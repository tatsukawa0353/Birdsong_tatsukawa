"""Issue #14: fixed-target Phase 0, preserving the legacy oscillator and learner.

Only training-goal randomization changes. Keep callback evaluations at their
legacy pre-update timing, and publish a separate post-final-update endpoint.
Never reuse or overwrite an existing experiment directory.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from zoneinfo import ZoneInfo

import numpy as np
from scipy.io import wavfile
import torch

from .legacy_paths import model_files, protected_path
from .acoustics import AcousticConfig, distance
from .experiments import EvaluationCallback, make_agent, phase0_factories, rollout, write_json
from .sine_env import SineConfig, SineEnv, oscillate


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs/issue14_fixed_target.json"
DEFAULT_OUTPUT = ROOT / "results/phase0_fixed_target_20261108"
WORKFLOW = "issue14_phase0_fixed_target"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_hash(agent):
    h = hashlib.sha256()
    for key, tensor in sorted(agent.policy.state_dict().items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        h.update(key.encode())
        h.update(array.dtype.str.encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def _state_digest(value):
    """Deterministic recursive hash of optimizer state, without torch ZIP metadata."""
    h = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            add(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            h.update(b"array" + array.dtype.str.encode() + str(array.shape).encode() + array.tobytes())
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                add(key)
                add(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode() + str(len(item)).encode())
            for element in item:
                add(element)
        else:
            h.update(type(item).__name__.encode() + repr(item).encode())

    add(value)
    return h.hexdigest()


def learner_signature(agent):
    optimizers = {}
    for name, parent, attribute in (("policy", agent.policy, "optimizer"),
                                    ("actor", getattr(agent, "actor", None), "optimizer"),
                                    ("critic", getattr(agent, "critic", None), "optimizer"),
                                    ("entropy", agent, "ent_coef_optimizer")):
        optimizer = getattr(parent, attribute, None)
        if optimizer is not None:
            optimizers[name] = optimizer.state_dict()
    result = dict(policy_sha256=parameter_hash(agent), num_timesteps=int(agent.num_timesteps),
                  gradient_updates=int(agent._n_updates), optimizer_sha256=_state_digest(optimizers),
                  rng_sha256=hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state()),
                                                        protocol=4) + torch.get_rng_state().numpy().tobytes()).hexdigest())
    for name in ("replay_buffer", "rollout_buffer"):
        buffer = getattr(agent, name, None)
        if buffer is not None:
            result[name] = dict(pos=int(buffer.pos), full=bool(buffer.full), size=int(buffer.size()))
            length = buffer.buffer_size if buffer.full else buffer.pos
            result[name]["content_sha256"] = _state_digest({key: value[:length] for key, value in vars(buffer).items()
                if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == buffer.buffer_size})
    result["entropy_sha256"] = _state_digest({key: getattr(agent, key, None) for key in ("log_ent_coef", "ent_coef_tensor")})
    result["action_rng_sha256"] = _state_digest(agent.action_space.np_random.bit_generator.state)
    envs = getattr(agent.get_env(), "envs", [])
    result["training_env_rng_sha256"] = _state_digest([e.unwrapped.np_random.bit_generator.state for e in envs])
    result["cuda_rng_sha256"] = _state_digest(torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else None
    return result


def tracking_metrics(target_controls, controls):
    errors = np.abs(np.asarray(controls) - np.asarray(target_controls))
    return dict(frequency_mae_hz=float(errors[:, 0].mean()),
                frequency_max_abs_hz=float(errors[:, 0].max()),
                amplitude_mae=float(errors[:, 1].mean()),
                amplitude_max_abs=float(errors[:, 1].max()))


def evaluate_agent(agent, factory, steps, phase):
    """A fresh deterministic 35-action episode; no learner/optimizer/buffer updates."""
    start = time.perf_counter()
    before = learner_signature(agent)
    env = factory()
    observations = []

    def policy(obs, _):
        observations.append(obs.copy())
        return agent.predict(obs, deterministic=True)[0]

    try:
        result, audio, actions = rollout(env, policy)
        controls = np.array([env.decode(action) for action in actions])
        target_controls = np.column_stack((env.frequencies, env.amplitudes))
        snapshot = dict(target=env.target_audio.copy(), audio=audio, actions=actions,
                        physical_controls=controls, target_controls=target_controls,
                        observations=np.asarray(observations), sample_rate=env.config.sample_rate,
                        step_rewards=np.asarray(env.rewards), time_s=np.arange(env.horizon)*env.config.control_s)
    finally:
        env.close()
    after = learner_signature(agent)
    if before != after:
        raise RuntimeError("Evaluation changed learner weights, optimizer, buffer, counters or RNG")
    record = dict(steps=int(steps), **result, gradient_updates=int(agent._n_updates),
                  evaluation_phase=phase, parameter_sha256=after["policy_sha256"],
                  learner_before=before, learner_after=after, learner_unchanged=True,
                  tracking=tracking_metrics(target_controls, controls),
                  negative_reward_sum=-float(np.sum(snapshot["step_rewards"])),
                  evaluation_wall_s=time.perf_counter()-start)
    return record, snapshot


class FixedEvaluationCallback(EvaluationCallback):
    def __init__(self, factory, interval, threshold, output=None):
        super().__init__(factory, interval, threshold)
        self.initial_snapshot = None
        self.actions, self.rewards, self.dones = [], [], []
        self.spectral_costs, self.envelope_costs = [], []
        self.output, self.best = output, None

    def evaluate(self):
        phase = "initial" if self.num_timesteps == 0 else "callback_pre_update"
        record, snapshot = evaluate_agent(self.model, self.factory, self.num_timesteps, phase)
        self.evaluation_steps += record["env_steps"]
        self.evaluation_seconds += record["evaluation_wall_s"]
        self.entries.append(record)
        if self.initial_snapshot is None:
            self.initial_snapshot = snapshot
        if self.best is None or record["error"] < self.best["error"]:
            self.best = record
            if self.output is not None:
                self.model.save(self.output/"best_agent.zip")
        if self.output is not None:
            write_json(self.output/"progress.json", dict(steps=self.num_timesteps, error=record["error"], best_error=self.best["error"]))

    def _on_step(self):
        for name in ("actions", "rewards", "new_obs"):
            if not np.isfinite(self.locals[name]).all():
                raise FloatingPointError(f"Non-finite training {name}")
        self.actions.append(np.asarray(self.locals["actions"][0]).copy())
        self.rewards.append(float(self.locals["rewards"][0]))
        self.dones.append(bool(self.locals["dones"][0]))
        self.spectral_costs.append(float(self.locals["infos"][0]["log_spectral"]))
        self.envelope_costs.append(float(self.locals["infos"][0]["envelope"]))
        return super()._on_step()


def effective_hyperparameters(agent):
    values = dict(policy=type(agent.policy).__name__, net_arch=agent.policy.net_arch,
                  activation_fn=agent.policy.activation_fn.__name__, device=str(agent.device),
                  seed=int(agent.seed), torch_threads=torch.get_num_threads())
    names = ("learning_rate", "gamma", "batch_size", "buffer_size", "learning_starts", "gradient_steps",
             "tau", "n_steps", "n_epochs", "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm",
             "normalize_advantage", "policy_delay", "target_policy_noise", "target_noise_clip",
             "target_update_interval", "target_entropy", "use_sde", "sde_sample_freq")
    for name in names:
        value = getattr(agent, name, None)
        if isinstance(value, (str, bool, int, float)):
            values[name] = value
    train_freq = getattr(agent, "train_freq", None)
    if train_freq is not None:
        values["train_freq"] = dict(frequency=int(train_freq.frequency), unit=train_freq.unit.value)
    for name in ("clip_range", "clip_range_vf"):
        value = getattr(agent, name, None)
        if value is not None:
            values[name] = float(value(1.0)) if callable(value) else value
    noise = getattr(agent, "action_noise", None)
    if noise is not None:
        values["action_noise"] = dict(type=type(noise).__name__, mean=noise._mu.tolist(), std=noise._sigma.tolist())
    if getattr(agent, "log_ent_coef", None) is not None:
        values["initial_entropy_coefficient"] = float(agent.log_ent_coef.detach().exp().cpu())
    optimizers = {}
    for name, parent in (("policy", agent.policy), ("actor", getattr(agent, "actor", None)),
                         ("critic", getattr(agent, "critic", None))):
        optimizer = getattr(parent, "optimizer", None)
        if optimizer is not None:
            optimizers[name] = [{key: value for key, value in group.items() if key != "params"}
                                for group in optimizer.state_dict()["param_groups"]]
    values["optimizer_param_groups"] = optimizers
    return values


def save_snapshot(directory, snapshot):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/"rollout.npz", **snapshot)
    rate = int(snapshot["sample_rate"])
    wavfile.write(directory/"target_raw.wav", rate, snapshot["target"].astype(np.float32))
    wavfile.write(directory/"generated_raw.wav", rate, snapshot["audio"].astype(np.float32))


def train_one(name, seed, settings, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train_factory, eval_factory = phase0_factories(settings)
    env = train_factory()
    agent = make_agent(name, env, seed, settings)
    effective = effective_hyperparameters(agent)
    callback = FixedEvaluationCallback(eval_factory, settings["eval_interval"], settings.get("threshold", .08), output)
    from stable_baselines3.common.logger import Logger
    numeric = {}
    class FiniteLogger(Logger):
        def record(self, key, value, exclude=None):
            if isinstance(value, (int, float, np.number)):
                if not np.isfinite(value):
                    raise FloatingPointError(f"Non-finite {key}: {value}")
                item = numeric.setdefault(key, dict(count=0, minimum=float(value), maximum=float(value)))
                item.update(count=item["count"]+1, minimum=min(item["minimum"], float(value)), maximum=max(item["maximum"], float(value)))
            super().record(key, value, exclude)
    agent.set_logger(FiniteLogger(None, []))
    optimizer_steps, hooks = {}, []
    seen = set()
    for label, parent in (("policy", agent.policy), ("actor", getattr(agent, "actor", None)),
                          ("critic", getattr(agent, "critic", None)), ("entropy", agent)):
        optimizer = getattr(parent, "ent_coef_optimizer" if label == "entropy" else "optimizer", None)
        if optimizer is None or id(optimizer) in seen:
            continue
        seen.add(id(optimizer))
        optimizer_steps[label] = 0
        def pre(opt, args, kwargs):
            for group in opt.param_groups:
                for param in group["params"]:
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise FloatingPointError("Non-finite gradient")
        def post(opt, args, kwargs, label=label):
            optimizer_steps[label] += 1
            for group in opt.param_groups:
                for param in group["params"]:
                    if not torch.isfinite(param).all():
                        raise FloatingPointError("Non-finite parameter")
        hooks.extend([optimizer.register_step_pre_hook(pre), optimizer.register_step_post_hook(post)])
    start = time.perf_counter()
    try:
        agent.learn(total_timesteps=settings["steps"], callback=callback)
        learn_wall = time.perf_counter()-start
        if agent.num_timesteps != settings["steps"]:
            raise RuntimeError("Actual training steps differ from the registered budget")
        final, snapshot = evaluate_agent(agent, eval_factory, agent.num_timesteps, "post_final_update")
        if final["error"] < callback.best["error"]:
            callback.best = final
            agent.save(output/"best_agent.zip")
        write_json(output/"best.json", dict(**callback.best, model_sha256=digest(output/"best_agent.zip"),
                   selection="minimum recorded in-target error, includes initial; selected on repeated target; not generalization"))
        initial = callback.entries[0]
        # Keep the pre-update callback trace, but the presentation endpoint must
        # be the same final policy as the summary, saved model, waveform and plots.
        curve = [entry for entry in callback.entries if entry["steps"] < agent.num_timesteps] + [final]
        save_snapshot(output/"initial", callback.initial_snapshot)
        save_snapshot(output, snapshot)
        agent.save(output/"agent")
        if parameter_hash(agent) != final["parameter_sha256"]:
            raise RuntimeError("Final model changed after its evaluation")
        np.savez_compressed(output/"training_trace.npz", actions=np.asarray(callback.actions),
                            rewards=np.asarray(callback.rewards), dones=np.asarray(callback.dones),
                            log_spectral=np.asarray(callback.spectral_costs), envelope=np.asarray(callback.envelope_costs))
        write_json(output/"curve.json", curve)
        write_json(output/"callback_curve.json", callback.entries)
        summary = dict(algorithm=name, seed=seed, requested_steps=settings["steps"],
                       training_steps=int(agent.num_timesteps), training_episodes=callback.training_episodes,
                       training_partial_episode_steps=int(agent.num_timesteps % env.horizon),
                       evaluation_steps=callback.evaluation_steps+final["env_steps"],
                       evaluation_episodes=len(callback.entries)+1,
                       training_generated_s=agent.num_timesteps*env.config.control_s,
                       evaluation_generated_s=(callback.evaluation_steps+final["env_steps"])*env.config.control_s,
                       training_wall_s=learn_wall-callback.evaluation_seconds,
                       evaluation_wall_s=callback.evaluation_seconds+final["evaluation_wall_s"],
                       total_wall_s=time.perf_counter()-start,
                       initial_error=initial["error"], initial_metrics=initial["metrics"],
                       initial_tracking=initial["tracking"], error=final["error"], metrics=final["metrics"],
                       tracking=final["tracking"], final_parameter_sha256=final["parameter_sha256"],
                       final_evaluation_phase="post_final_update", effective_hyperparameters=effective,
                       initial_record=initial, final_record=final, failed=final["failed"],
                       training_numerical_failures=callback.training_numerical_failures,
                       model_sha256=digest(output/"agent.zip"), rollout_sha256=digest(output/"rollout.npz"),
                       optimizer_steps=optimizer_steps, sb3_n_updates=int(agent._n_updates),
                       update_counter_meaning="PPO epochs; TD3/SAC gradient iterations; see optimizer_steps for actual optimizer calls",
                       nonfinite_detected=0, numeric_training_metrics=numeric,
                       training_reward_terms=settings.get("reward_terms", "combined"),
                       evaluation_reward_terms="combined",
                       best_error=callback.best["error"], best_steps=callback.best["steps"],
                       first_threshold_step=next((e["steps"] for e in curve if e["error"] <= settings.get("threshold", .08)), None),
                       interpretation="in-target deterministic evaluation; not held-out generalization")
        write_json(output/"summary.json", summary)
        print(json.dumps(dict(algorithm=name, seed=seed, initial=initial["error"], final=final["error"],
                              tracking=final["tracking"], training_steps=agent.num_timesteps)), flush=True)
        return summary
    finally:
        for hook in hooks:
            hook.remove()
        env.close()


def protected_files():
    excluded = {".venv", ".venv-rl", ".issue14-work", "venv", "build", "__pycache__", ".pytest_cache", "private_runs", "sample_song"}
    result = {}
    for base, directories, names in os.walk(ROOT):
        directories[:] = [name for name in directories if name not in excluded]
        if Path(base) == ROOT/"results":
            directories[:] = [name for name in directories if name != DEFAULT_OUTPUT.name]
        for name in names:
            path = Path(base)/name
            # New issue-specific reporting/auditing code may be completed while
            # training runs; computational sources are separately frozen below.
            if name.startswith("issue14") or name in ("test_issue14.py", "ISSUE14_REPORT.md"):
                continue
            if path.is_file():
                result[str(path.relative_to(ROOT.parent))] = digest(path)
    for path in model_files():
        if path.is_file():
            result[path.name] = digest(path)
    return result


def verify_protected(output):
    before = json.loads((output/"protected_before.json").read_text())
    changed = [relative for relative, expected in before.items()
               if not protected_path(relative).is_file() or digest(protected_path(relative)) != expected]
    result = dict(passed=not changed, files_checked=len(before), changed=changed)
    write_json(output/"preservation_check.json", result)
    if changed:
        raise RuntimeError(f"Existing artifacts changed: {changed}")
    return result


def diagnostics(output, settings):
    training_factory, evaluation_factory = phase0_factories(settings)
    reference_env = evaluation_factory()
    reference_env.reset(seed=10000)
    reference = reference_env.target_audio.copy()
    target_controls = np.column_stack((reference_env.frequencies, reference_env.amplitudes))
    checks = []
    for label, factory in (("training", training_factory), ("evaluation", evaluation_factory)):
        for seed in (0, 1, 2, 10000):
            env = factory()
            for episode_index in range(3):
                env.reset(seed=seed if episode_index == 0 else None)
                same = np.array_equal(env.target_audio, reference)
                controls_equal = np.array_equal(np.column_stack((env.frequencies, env.amplitudes)), target_controls)
                checks.append(dict(environment=label, seed=seed, episode=episode_index,
                                   target_bitwise_equal=bool(same), controls_bitwise_equal=bool(controls_equal),
                                   phase=env.phase, index=env.index,
                                   previous_zero=bool(np.all(env.previous == 0))))
            env.close()
    env = evaluation_factory()
    result, audio, actions = rollout(env, lambda obs, e: e.goal_actions[e.index])
    applied = np.array([env.decode(a) for a in actions])
    snapshot = dict(target=reference, audio=audio, actions=actions, physical_controls=applied,
                    target_controls=target_controls, sample_rate=env.config.sample_rate,
                    step_rewards=np.asarray(env.rewards), time_s=np.arange(env.horizon)*env.config.control_s)
    direct, phase = [], 0.0
    for frequency, amplitude in target_controls:
        block, phase = oscillate(frequency, amplitude, phase, env.block, env.config.sample_rate)
        direct.append(block)
    direct_audio = np.concatenate(direct)
    result.update(max_waveform_abs_error=float(np.max(np.abs(audio-reference))),
                  negative_reward_sum=-float(np.sum(env.rewards)),
                  direct_float64_series_error=distance(reference, direct_audio, env.config.sample_rate, scale=1.0)["total"],
                  direct_float64_bitwise_equal=bool(np.array_equal(reference, direct_audio)),
                  note="Action Oracle uses unchanged float32 encoding; direct f/A replay is separate")
    passed = (all(c["target_bitwise_equal"] and c["controls_bitwise_equal"] and c["phase"] == 0 and
                  c["index"] == 0 and c["previous_zero"] for c in checks) and result["error"] < 1e-5 and
              result["direct_float64_bitwise_equal"] and env.horizon == 35 and env.block == 480)
    (output/"diagnostics").mkdir()
    save_snapshot(output/"diagnostics", snapshot)
    write_json(output/"diagnostics/oracle.json", dict(passed=bool(passed), **result))
    write_json(output/"diagnostics/target_identity.json", dict(passed=bool(passed), checks=checks))
    env.close()
    reference_env.close()
    if not passed:
        raise RuntimeError("Target/Oracle diagnostic gate failed; do not train")
    return result


def run(output, settings, workers=1):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new experiment ID; existing results are never overwritten")
    legacy = json.loads((ROOT/"configs/phase0.json").read_text())
    if (any(settings.get(key) != value for key, value in legacy.items() if key not in ("steps", "eval_interval")) or
            (settings.get("steps"), settings.get("eval_interval")) not in ((4096, 512), (102400, 2048)) or
            settings.get("randomize_training_target") is not False):
        raise ValueError("Only the registered 4096/512 or 102400/2048 Phase 0 protocol is allowed")
    preserved = protected_files()
    output.mkdir(parents=True)
    write_json(output/"protected_before.json", preserved)
    write_json(output/"config.json", settings)
    sources = ("issue14.py", "sine_env.py", "experiments.py", "acoustics.py", "plotting.py",
               "configs/phase0.json", "configs/issue14_fixed_target.json", "requirements-lock.txt")
    source_hashes = {name: digest(ROOT/name) for name in sources}
    (output/"source_changes.patch").write_text(subprocess.check_output(["git", "diff", "HEAD", "--", *sources], cwd=ROOT, text=True))
    oracle = diagnostics(output, settings)
    manifest = dict(workflow=WORKFLOW, issue_url="https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/14",
                    created_utc=datetime.now(timezone.utc).isoformat(), client_date_jst=datetime.now(ZoneInfo("Asia/Tokyo")).date().isoformat(),
                    settings=settings, environment=asdict(SineConfig(randomize=False, reseed_action_space_on_reset=settings.get("reseed_action_space_on_reset", True))),
                    acoustic_config=asdict(AcousticConfig()), acoustic_scale=1.0,
                    source_hashes=source_hashes, command=" ".join(sys.argv),
                    reproducible_command=f"PYTHONDONTWRITEBYTECODE=1 Deeplearning/.venv-rl/bin/python -m Deeplearning.issue14 --replace --workers {workers}",
                    base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), workers=workers,
                    target_frequency_hz=[1000, 2000], target_amplitude="0.6*(0.8+0.2*cos(2*pi*k/34)); not constant",
                    duration_s=.35, control_s=.01, sample_rate=48000, horizon=35,
                    phase="reset to 0 per episode; continuous between 480-sample blocks",
                    training_budget=f"{settings['steps']} actions = {settings['steps']//35} completed episodes and {settings['steps']%35} partial steps",
                    evaluation_schedule=f"0 plus each {settings['eval_interval']} callback pre-update; independent final post-update",
                    presentation_curve="Replace the final callback entry with the post-update evaluation; retain callback_curve.json",
                    reward="negative legacy distance on the corresponding 10 ms block; smooth_weight=0",
                    primary="legacy distance on the full 0.35 s waveform; not a sum of block errors",
                    evaluation="in-target deterministic evaluation, not held-out or unknown-target generalization",
                    oracle_error=oracle["error"], old_reference="Deeplearning/results/phase0",
                    only_training_condition_change="fixed target retained; registered budget and evaluation interval extended; explicit-only action-space reseeding is an independent reproducibility change",
                    other_changes=["retain evaluation timing and weights/buffer/optimizer/RNG audit",
                                   "presentation endpoint uses final updated policy", "retain initial/final tracking and raw audio"],
                    reproducibility_limit="New runs seed action space on explicit reset only; legacy defaults/results untouched. Old 4096 values are a different RNG-management reference, not a budget-only causal comparison.",
                    package_versions={name: importlib.metadata.version(name) for name in
                                      ("numpy", "scipy", "torch", "stable-baselines3", "gymnasium", "matplotlib", "SciencePlots")})
    write_json(output/"manifest.json", manifest)
    started = time.perf_counter()
    summaries = []
    if workers == 1:
        for seed in settings["seeds"]:
            for name in settings["algorithms"]:
                summaries.append(train_one(name, seed, settings, output/f"{name}_{seed}"))
                write_json(output/"summary.json", summaries)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(train_one, name, seed, settings, output/f"{name}_{seed}")
                       for seed in settings["seeds"] for name in settings["algorithms"]]
            for future in as_completed(futures):
                summaries.append(future.result())
                write_json(output/"summary.json", summaries)
    write_json(output/"execution.json", dict(completed_runs=len(summaries), expected_runs=9,
               execution_wall_s=time.perf_counter()-started,
               training_steps=sum(s["training_steps"] for s in summaries),
               evaluation_steps=sum(s["evaluation_steps"] for s in summaries),
               training_generated_s=sum(s["training_generated_s"] for s in summaries),
               evaluation_generated_s=sum(s["evaluation_generated_s"] for s in summaries)))
    verify_protected(output)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--replace", action="store_true", help="Stage, audit, then replace only the canonical Issue 14 results")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    if args.replace:
        from .issue14_transaction import replace_results
        replace_results(settings, args.workers)
    else:
        run(args.output, settings, args.workers)


if __name__ == "__main__":
    main()
