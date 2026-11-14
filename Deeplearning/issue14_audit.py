"""Independently verify saved fixed-target Phase 0 results, without learning.

Only lightweight sine-wave replay and saved-audio metric recomputation occur.
No historical outputs are written and the native syrinx model is never loaded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import subprocess

import numpy as np
from scipy.io import wavfile

from .legacy_paths import protected_path, legacy_phase0_root
from .acoustics import AcousticConfig, distance
from .sine_env import SineConfig, SineEnv


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results/phase0_fixed_target_20261108"
ALGORITHMS = ("PPO", "TD3", "SAC")
SEEDS = (0, 1, 2)


def policy_sha256(agent):
    """Independent implementation of the recorded tensor-hash protocol."""
    digest = hashlib.sha256()
    for key, value in sorted(agent.policy.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(key.encode())
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


class Auditor:
    def __init__(self, output):
        self.output = Path(output).resolve()
        settings = json.loads((self.output/"config.json").read_text())
        self.steps, self.interval = settings["steps"], settings["eval_interval"]
        self.schedule = list(range(0, self.steps+1, self.interval))
        integration = self.output/"reward_ablation/manifest.json"
        self.historical_commit = json.loads(integration.read_text())["control_commit"] if integration.exists() else None
        self.historical_source_checks = []
        self.errors = []
        self.checks = 0
        self.counts = dict(runs_expected=9, runs_found=0, curve_points=0,
                           callback_points=0, rollout_archives=0,
                           replayed_sine_steps=0, policy_reload_episodes=0,
                           audio_metric_recomputations=0, protected_files=0,
                           source_hashes=0, learner_unchanged_records=0,
                           training_steps=0, evaluation_steps=0,
                           training_trace_steps=0, raw_wav_files=0)
        self.final_timing_comparisons = []
        self.summaries = {}
        env = SineEnv(SineConfig(randomize=False, target="glide"))
        env.reset(seed=10000)
        self.target = env.target_audio.copy()
        self.target_controls = np.column_stack((env.frequencies, env.amplitudes))
        env.close()

    def check(self, passed, code, path, expected=None, actual=None):
        self.checks += 1
        if not passed:
            path = Path(path)
            relative = str(path.relative_to(self.output)) if path.is_relative_to(self.output) else str(path)
            self.errors.append(_plain(dict(code=code, path=relative, expected=expected, actual=actual)))
        return bool(passed)

    def scalar(self, actual, expected, path, *, atol=1e-12, rtol=1e-9):
        try:
            actual, expected = float(actual), float(expected)
            passed = np.isfinite(actual) and np.isfinite(expected) and abs(actual-expected) <= atol+rtol*abs(expected)
        except (TypeError, ValueError, OverflowError):
            passed = False
        return self.check(passed, "numeric_mismatch", path, expected, actual)

    def exact_array(self, actual, expected, path):
        actual, expected = np.asarray(actual), np.asarray(expected)
        return self.check(actual.shape == expected.shape and np.array_equal(actual, expected),
                          "array_mismatch", path,
                          dict(shape=expected.shape, values="exact equality"),
                          dict(shape=actual.shape, equal=bool(actual.shape == expected.shape and np.array_equal(actual, expected))))

    def read_json(self, path):
        try:
            value = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            self.check(False, "json_read", path, "readable JSON", str(exc))
            return None
        self.check(True, "json_read", path)
        return value

    def metrics(self, actual, expected, path):
        if not self.check(isinstance(actual, dict), "metric_schema", path, "dict", type(actual).__name__):
            return
        for key, value in expected.items():
            if isinstance(value, bool):
                self.check(actual.get(key) is value, "metric_flag", Path(path) / key, value, actual.get(key))
            else:
                self.scalar(actual.get(key), value, Path(path) / key)

    @staticmethod
    def tracking(controls, target):
        differences = np.abs(controls-target)
        return dict(frequency_mae_hz=float(np.mean(differences[:, 0])),
                    frequency_max_abs_hz=float(np.max(differences[:, 0])),
                    amplitude_mae=float(np.mean(differences[:, 1])),
                    amplitude_max_abs=float(np.max(differences[:, 1])))

    def eval_record(self, record, path, algorithm):
        if not self.check(isinstance(record, dict), "evaluation_schema", path, "dict", type(record).__name__):
            return
        self.scalar(record.get("error"), record.get("metrics", {}).get("total"), Path(path) / "error")
        self.scalar(record.get("env_steps"), 35, Path(path) / "env_steps")
        before, after = record.get("learner_before"), record.get("learner_after")
        valid = isinstance(before, dict) and isinstance(after, dict)
        self.check(valid, "learner_state_schema", path, "before/after state digests", None)
        if valid:
            required = {"policy_sha256", "num_timesteps", "gradient_updates", "optimizer_sha256", "rng_sha256"}
            buffer_key = "rollout_buffer" if algorithm == "PPO" else "replay_buffer"
            for label, signature in (("before", before), ("after", after)):
                self.check(required | {buffer_key} <= set(signature), "learner_signature_fields", Path(path) / label,
                           sorted(required | {buffer_key}), sorted(signature))
                for key in ("policy_sha256", "optimizer_sha256", "rng_sha256"):
                    value = signature.get(key)
                    valid_hash = isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
                    self.check(valid_hash, "learner_signature_sha256", Path(path) / label / key, "64 lowercase hex characters", value)
                buffer = signature.get(buffer_key)
                if self.check(isinstance(buffer, dict), "learner_buffer_schema", Path(path) / label, "buffer dict", buffer):
                    capacity = 256 if algorithm == "PPO" else 50000
                    valid_buffer = (isinstance(buffer.get("pos"), int) and 0 <= buffer["pos"] <= capacity and
                                    isinstance(buffer.get("size"), int) and 0 <= buffer["size"] <= capacity and
                                    isinstance(buffer.get("full"), bool))
                    self.check(valid_buffer, "learner_buffer_fields", Path(path) / label / buffer_key,
                               "finite integer pos/size within capacity and bool full", buffer)
            self.check(before == after and record.get("learner_unchanged") is True,
                       "evaluation_changed_learner", path, "exact unchanged learner state", dict(before=before, after=after))
            self.check(record.get("parameter_sha256") == after.get("policy_sha256"),
                       "evaluation_parameter_hash", path, after.get("policy_sha256"), record.get("parameter_sha256"))
            self.scalar(record.get("gradient_updates"), after.get("gradient_updates"), Path(path) / "gradient_updates")
            self.scalar(record.get("steps"), after.get("num_timesteps"), Path(path) / "steps")
            self.counts["learner_unchanged_records"] += 1

    def rollout(self, path, record, *, agent=None):
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in
                      ("target", "audio", "actions", "physical_controls", "target_controls", "step_rewards")}
            rate = int(archive["sample_rate"])
        self.counts["rollout_archives"] += 1
        valid = True
        for key, shape, dtype in (("target", (16800,), np.float64), ("audio", (16800,), np.float64),
                                  ("actions", (35, 2), np.float32), ("physical_controls", (35, 2), np.float64),
                                  ("target_controls", (35, 2), np.float64), ("step_rewards", (35,), np.float64)):
            value = arrays[key]
            valid &= self.check(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(),
                                "rollout_array_schema", Path(path) / key,
                                dict(shape=shape, dtype=str(np.dtype(dtype)), finite=True),
                                dict(shape=value.shape, dtype=str(value.dtype), finite=bool(np.isfinite(value).all())))
        self.scalar(rate, 48000, Path(path) / "sample_rate")
        self.exact_array(arrays["target"], self.target, Path(path) / "target")
        self.exact_array(arrays["target_controls"], self.target_controls, Path(path) / "target_controls")
        for filename, values in (("target_raw.wav", arrays["target"]), ("generated_raw.wav", arrays["audio"])):
            wav_path = Path(path).parent / filename
            wav_rate, wave = wavfile.read(wav_path)
            expected = values.astype(np.float32)
            same = wave.dtype == np.float32 and wav_rate == 48000 and wave.shape == expected.shape and wave.tobytes() == expected.tobytes()
            self.check(same, "raw_wav_floatcast", wav_path, "48000Hz float32 exact NPZ cast, no gain normalization", dict(rate=wav_rate, dtype=str(wave.dtype), exact=bool(same)))
            self.counts["raw_wav_files"] += 1
        if not valid:
            return
        self.check(np.all(abs(arrays["actions"]) <= 1), "action_bounds", Path(path) / "actions", "[-1,1]", None)
        env = SineEnv(SineConfig(randomize=False, target="glide"))
        try:
            observation, _ = env.reset(seed=10000)
            for index, saved_action in enumerate(arrays["actions"]):
                if agent is not None:
                    predicted, _ = agent.predict(observation, deterministic=True)
                    self.exact_array(predicted, saved_action, Path(path) / f"predicted_action_{index}")
                observation, reward, terminated, truncated, _ = env.step(saved_action)
                self.check(terminated == (index == 34) and not truncated,
                           "episode_lifecycle", Path(path) / str(index), index == 34, terminated)
                self.scalar(reward, arrays["step_rewards"][index], Path(path) / f"step_reward_{index}")
            self.exact_array(np.concatenate(env.audio), arrays["audio"], Path(path) / "phase_continuous_replay")
            self.exact_array(np.asarray([env.decode(action) for action in arrays["actions"]]), arrays["physical_controls"], Path(path) / "physical_controls")
            self.counts["replayed_sine_steps"] += 35
            self.counts["policy_reload_episodes"] += int(agent is not None)
        finally:
            env.close()
        recomputed = distance(arrays["target"], arrays["audio"], rate, scale=1.)
        self.metrics(record.get("metrics"), recomputed, path)
        self.scalar(record.get("error"), recomputed["total"], Path(path) / "full_error")
        self.metrics(record.get("tracking"), self.tracking(arrays["physical_controls"], self.target_controls), Path(path) / "tracking")
        if "negative_reward_sum" in record:
            self.scalar(record["negative_reward_sum"], -float(np.sum(arrays["step_rewards"])), Path(path) / "negative_reward_sum")
        self.counts["audio_metric_recomputations"] += 1

    def hashes(self, mapping, base, path, *, protected=False):
        if not self.check(isinstance(mapping, dict) and bool(mapping), "hash_schema", path, "nonempty relative SHA256 mapping", type(mapping).__name__):
            return
        base = Path(base).resolve()
        for name, expected in mapping.items():
            destination = (protected_path(name) if protected and base == ROOT.parent else base / name).resolve()
            if not self.check(not Path(name).is_absolute() and destination.is_relative_to(base), "hash_scope", path, "path inside repository", name):
                continue
            try:
                actual = hashlib.sha256(destination.read_bytes()).hexdigest()
                historical = False
                # Result bytes must still agree at the relocated path. Source
                # and documentation hashes can instead verify the exact Git
                # version used by the frozen A experiment after later work.
                if actual != expected and self.historical_commit and not destination.is_relative_to(ROOT/"results"):
                    name_at_commit = str(destination.relative_to(ROOT.parent))
                    saved = subprocess.run(["git","show",f"{self.historical_commit}:{name_at_commit}"],cwd=ROOT,capture_output=True)
                    historical = saved.returncode == 0 and hashlib.sha256(saved.stdout).hexdigest() == expected
                    if historical:
                        self.historical_source_checks.append(dict(path=name_at_commit, commit=self.historical_commit, sha256=expected))
                self.check(actual == expected or historical, "protected_changed" if protected else "source_changed", destination, expected, actual)
                self.counts["protected_files" if protected else "source_hashes"] += 1
            except OSError as exc:
                self.check(False, "hash_read", destination, expected, str(exc))

    def run(self):
        start = time.perf_counter()
        try:
            self._run()
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, RuntimeError) as exc:
            self.check(False, "schema_or_io_error", self.output, "complete valid experiment", f"{type(exc).__name__}: {exc}")
        return dict(passed=not self.errors, checks=self.checks, errors=self.errors,
                    historical_source_checks=self.historical_source_checks,
                    counts=self.counts, wall_s=time.perf_counter()-start,
                    callback_vs_post_final=self.final_timing_comparisons,
                    audited_utc=datetime.now(timezone.utc).isoformat(),
                    scope="Saved fixed-target Phase 0 artifacts; sine-only replay and saved-audio metric recomputation, no training or syrinx integration.",
                    limitations=["Intermediate unsaved audio cannot be independently recomputed; its metric identity and recorded learner-state invariance are checked.",
                                 "Evaluation learner-state invariance is based on recorded before/after digests and separately tested implementation."])

    def _run(self):
        manifest_path = self.output / "manifest.json"
        manifest = self.read_json(manifest_path)
        if not self.check(isinstance(manifest, dict), "manifest_schema", manifest_path, "JSON object", type(manifest).__name__):
            return
        if not self.check(manifest.get("workflow") == "issue14_phase0_fixed_target", "workflow_identity", manifest_path,
                          "issue14_phase0_fixed_target", manifest.get("workflow")):
            return
        settings = manifest["settings"]
        self.check(settings == self.read_json(self.output / "config.json"), "copied_configuration", manifest_path, "matching config.json", None)
        self.check(settings == self.read_json(ROOT / "configs/issue14_fixed_target.json"), "frozen_configuration", manifest_path, "matching source config", None)
        legacy = self.read_json(ROOT / "configs/phase0.json")
        for key, value in legacy.items():
            if key in ("steps", "eval_interval"):
                continue
            self.check(settings.get(key) == value, "legacy_hyperparameter", manifest_path / key, value, settings.get(key))
        self.check(settings.get("randomize_training_target") is False, "fixed_training_target", manifest_path, False, settings.get("randomize_training_target"))
        if self.steps == 102400:
            self.check(settings.get("reseed_action_space_on_reset") is False, "explicit_seed_only", manifest_path, False, settings.get("reseed_action_space_on_reset"))
        expected_environment = asdict(SineConfig(randomize=False, reseed_action_space_on_reset=settings.get("reseed_action_space_on_reset", True)))
        recorded_environment=dict(manifest.get("environment",{}))
        recorded_environment.setdefault("reward_terms","combined")
        self.check(recorded_environment == expected_environment, "environment_configuration", manifest_path, expected_environment, recorded_environment)
        self.check(manifest.get("acoustic_config") == asdict(AcousticConfig()), "acoustic_configuration", manifest_path, asdict(AcousticConfig()), manifest.get("acoustic_config"))
        self.scalar(manifest.get("acoustic_scale"), 1., manifest_path / "acoustic_scale")
        self.hashes(manifest.get("source_hashes"), ROOT, manifest_path)
        if (self.output/"source_snapshot").is_dir():
            self.hashes(manifest.get("source_hashes"), self.output / "source_snapshot", manifest_path / "source_snapshot")
        before = self.read_json(self.output / "protected_before.json")
        self.hashes(before, ROOT.parent, self.output / "protected_before.json", protected=True)
        preservation = self.read_json(self.output / "preservation_check.json")
        if self.check(isinstance(preservation, dict), "preservation_schema", self.output / "preservation_check.json", "dict", type(preservation).__name__):
            self.check(preservation.get("passed") is True and preservation.get("changed") == [], "runner_preservation", self.output / "preservation_check.json", "PASS changed=[]", preservation)
            self.scalar(preservation.get("files_checked"), len(before), self.output / "preservation_check.json" / "files_checked")
        expected = {self.output / f"{name}_{seed}" / "summary.json": (name, seed) for name in ALGORITHMS for seed in SEEDS}
        found = set(self.output.glob("*_[012]/summary.json"))
        self.counts["runs_found"] = len(found)
        self.check(found == set(expected), "complete_run_set", self.output, 9, len(found))
        for path, (name, seed) in expected.items():
            try:
                self.audit_run(path, name, seed)
            except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, RuntimeError) as exc:
                self.check(False, "run_schema_or_io_error", path, "valid completed run", f"{type(exc).__name__}: {exc}")
        aggregate_path = self.output / "summary.json"
        aggregate = self.read_json(aggregate_path)
        if self.check(isinstance(aggregate, list), "aggregate_summary_schema", aggregate_path, "list of nine summaries", type(aggregate).__name__):
            mapping = {(record.get("algorithm"), record.get("seed")): record for record in aggregate}
            self.check(len(aggregate) == 9 and mapping == self.summaries, "aggregate_summary_identity", aggregate_path, "exact nine individual run summaries", None)
        with np.load(legacy_phase0_root()/"PPO_0/rollout.npz", allow_pickle=False) as archive:
            self.exact_array(archive["target"], self.target, self.output / "historical_evaluation_target")
        self.oracle()
        execution_path = self.output / "execution.json"
        execution = self.read_json(execution_path)
        if self.check(isinstance(execution, dict), "execution_schema", execution_path, "dict", type(execution).__name__):
            evaluations = (len(self.schedule)+1)*35
            for key, value in dict(completed_runs=9, expected_runs=9, training_steps=9*self.steps,
                                   evaluation_steps=9*evaluations, training_generated_s=9*self.steps*.01,
                                   evaluation_generated_s=9*evaluations*.01).items():
                self.scalar(execution.get(key), value, execution_path / key)

    def audit_run(self, path, name, seed):
        summary = self.read_json(path)
        curve_path = path.parent / "curve.json"
        callback_path = path.parent / "callback_curve.json"
        curve, callback = self.read_json(curve_path), self.read_json(callback_path)
        if not self.check(isinstance(summary, dict) and isinstance(curve, list) and isinstance(callback, list), "run_schema", path, "summary and two curves", None):
            return
        self.check((summary.get("algorithm"), summary.get("seed")) == (name, seed), "run_identity", path, (name, seed), (summary.get("algorithm"), summary.get("seed")))
        self.summaries[(name, seed)] = summary
        evaluations = (len(self.schedule)+1)*35
        wanted = dict(training_steps=self.steps, requested_steps=self.steps, training_episodes=self.steps//35,
                      training_partial_episode_steps=self.steps%35, evaluation_steps=evaluations,
                      evaluation_episodes=len(self.schedule)+1, training_generated_s=self.steps*.01,
                      evaluation_generated_s=evaluations*.01, training_numerical_failures=0)
        for key, value in wanted.items():
            self.scalar(summary.get(key), value, path / key)
        self.check(summary.get("failed") is False and summary.get("final_evaluation_phase") == "post_final_update", "final_identity", path, "successful post_final_update", None)
        self.counts["training_steps"] += summary["training_steps"]
        self.counts["evaluation_steps"] += summary["evaluation_steps"]
        for entries, source, presentation in ((curve, curve_path, True), (callback, callback_path, False)):
            self.check([entry.get("steps") for entry in entries] == self.schedule, "evaluation_schedule", source, self.schedule, [entry.get("steps") for entry in entries])
            self.counts["curve_points" if presentation else "callback_points"] += len(entries)
            for entry in entries:
                n = entry["steps"]
                phase = ("initial" if n == 0 else "post_final_update" if presentation and n == self.steps else "callback_pre_update")
                self.check(entry.get("evaluation_phase") == phase, "evaluation_timing", source / str(n), phase, entry.get("evaluation_phase"))
                updates = (self.steps//256*5 if n == self.steps and phase == "post_final_update" else max(0, (n-1)//256)*5) if name == "PPO" else max(0, n-256-(phase != "post_final_update"))
                self.scalar(entry.get("gradient_updates"), updates, source / str(n) / "gradient_updates")
                self.eval_record(entry, source / str(n), name)
        self.check(curve[:-1] == callback[:-1], "retained_intermediate_evaluations", curve_path, "identical callback entries through step3584", None)
        self.check(summary.get("initial_record") == curve[0] and summary.get("final_record") == curve[-1], "summary_evaluation_records", path, "same curve initial/final records", None)
        for actual, expected, label in ((summary.get("initial_error"), curve[0]["error"], "initial_error"),
                                         (summary.get("error"), curve[-1]["error"], "error")):
            self.scalar(actual, expected, path / label)
        self.metrics(summary.get("initial_metrics"), curve[0]["metrics"], path / "initial_metrics")
        self.metrics(summary.get("metrics"), curve[-1]["metrics"], path / "metrics")
        self.metrics(summary.get("initial_tracking"), curve[0]["tracking"], path / "initial_tracking")
        self.metrics(summary.get("tracking"), curve[-1]["tracking"], path / "tracking")
        self.check(summary.get("final_parameter_sha256") == curve[-1]["parameter_sha256"], "final_parameter_identity", path, curve[-1]["parameter_sha256"], summary.get("final_parameter_sha256"))
        for filename, key in (("agent.zip", "model_sha256"), ("rollout.npz", "rollout_sha256")):
            actual = hashlib.sha256((path.parent / filename).read_bytes()).hexdigest()
            self.check(summary.get(key) == actual, "final_artifact_hash", path.parent / filename, summary.get(key), actual)
        self.rollout(path.parent / "initial/rollout.npz", curve[0])
        import torch
        from stable_baselines3 import PPO, SAC, TD3
        torch.set_num_threads(1)
        agent = {"PPO": PPO, "TD3": TD3, "SAC": SAC}[name].load(path.parent / "agent.zip", device="cpu")
        self.check(policy_sha256(agent) == summary["final_parameter_sha256"], "reloaded_parameter_hash", path.parent / "agent.zip", summary["final_parameter_sha256"], policy_sha256(agent))
        updates = self.steps//256*5 if name == "PPO" else max(0, self.steps-256)
        self.scalar(agent.num_timesteps, self.steps, path.parent / "agent.zip" / "num_timesteps")
        self.scalar(agent._n_updates, updates, path.parent / "agent.zip" / "gradient_updates")
        self.effective_settings(summary.get("effective_hyperparameters"), agent, name, seed, path)
        self.rollout(path.parent / "rollout.npz", curve[-1], agent=agent)
        self.training_trace(path.parent / "training_trace.npz")
        optimizers = {"policy": updates*4} if name == "PPO" else {"actor": updates//2 if name == "TD3" else updates, "critic": updates}
        if name == "SAC":
            optimizers["entropy"] = updates
        self.check(summary.get("optimizer_steps") == optimizers, "actual_optimizer_counts", path, optimizers, summary.get("optimizer_steps"))
        best = self.read_json(path.parent/"best.json")
        expected_best = min(callback+[curve[-1]], key=lambda entry: entry["error"])
        self.scalar(best["error"], expected_best["error"], path/"best_error")
        self.scalar(best["steps"], expected_best["steps"], path/"best_steps")
        best_agent = {"PPO": PPO, "TD3": TD3, "SAC": SAC}[name].load(path.parent/"best_agent.zip", device="cpu")
        self.check(policy_sha256(best_agent) == best["parameter_sha256"], "best_policy_hash", path, best["parameter_sha256"], policy_sha256(best_agent))
        self.check(summary.get("nonfinite_detected") == 0, "nonfinite_training", path, 0, summary.get("nonfinite_detected"))
        self.final_timing_comparisons.append(dict(algorithm=name, seed=seed,
            callback_error=callback[-1]["error"], post_final_error=curve[-1]["error"],
            parameter_hash_changed=callback[-1]["parameter_sha256"] != curve[-1]["parameter_sha256"],
            callback_gradient_updates=callback[-1]["gradient_updates"], final_gradient_updates=curve[-1]["gradient_updates"]))

    def effective_settings(self, values, agent, name, seed, path):
        if not self.check(isinstance(values, dict), "effective_settings_schema", path, "dict", type(values).__name__):
            return
        expected = dict(learning_rate=.0003, gamma=.99, batch_size=64, seed=seed, torch_threads=1)
        if name == "PPO":
            expected.update(n_steps=256, n_epochs=5, gae_lambda=.95, ent_coef=0., vf_coef=.5, max_grad_norm=.5)
        else:
            expected.update(buffer_size=50000, learning_starts=256, gradient_steps=1, tau=.005)
        if name == "TD3":
            expected.update(policy_delay=2, target_policy_noise=.2, target_noise_clip=.5)
        if name == "SAC":
            expected.update(target_update_interval=1, target_entropy=-2.)
        for key, value in expected.items():
            self.scalar(values.get(key), value, path / "effective_hyperparameters" / key)
            if key not in ("torch_threads",):
                self.scalar(getattr(agent, key), value, path / "reloaded_hyperparameters" / key)
        self.check(values.get("net_arch") == [32, 32] and agent.policy.net_arch == [32, 32],
                   "network_architecture", path, [32, 32], values.get("net_arch"))
        if name != "PPO":
            wanted = dict(frequency=1, unit="step")
            self.check(values.get("train_freq") == wanted and agent.train_freq.frequency == 1 and agent.train_freq.unit.value == "step",
                       "off_policy_update_frequency", path, wanted, values.get("train_freq"))
        if name == "TD3":
            noise = values.get("action_noise", {})
            self.check(noise.get("type") == "NormalActionNoise", "td3_noise_type", path, "NormalActionNoise", noise.get("type"))
            self.exact_array(noise.get("mean"), [0., 0.], path / "action_noise_mean")
            self.exact_array(noise.get("std"), [.1, .1], path / "action_noise_std")
        if name == "PPO":
            self.scalar(values.get("clip_range"), .2, path / "clip_range")
            self.scalar(agent.clip_range(1.), .2, path / "reloaded_clip_range")
            self.check(values.get("normalize_advantage") is True and agent.normalize_advantage is True,
                       "ppo_advantage_normalization", path, True, values.get("normalize_advantage"))
        if name == "SAC":
            self.check(values.get("ent_coef") == "auto" and agent.ent_coef == "auto", "sac_entropy_optimization", path, "auto", values.get("ent_coef"))
            self.scalar(values.get("initial_entropy_coefficient"), 1., path / "initial_entropy_coefficient")

    def training_trace(self, path):
        with np.load(path, allow_pickle=False) as archive:
            actions, rewards, dones = archive["actions"], archive["rewards"], archive["dones"]
        valid = self.check(actions.shape == (self.steps, 2) and rewards.shape == dones.shape == (self.steps,) and
                           np.isfinite(actions).all() and np.isfinite(rewards).all(),
                           "training_trace_schema", path, f"{self.steps} finite action/reward/done records", None)
        if valid:
            expected_dones = (np.arange(self.steps)+1) % 35 == 0
            self.exact_array(dones, expected_dones, path / "35_step_episodes")
            self.scalar(int(np.count_nonzero(dones)), self.steps//35, path / "completed_episodes")
            self.counts["training_trace_steps"] += len(actions)

    def oracle(self):
        path = self.output / "diagnostics/oracle.json"
        oracle = self.read_json(path)
        self.check(oracle.get("passed") is True, "oracle_gate", path, True, oracle.get("passed"))
        self.check(0 <= oracle["error"] < 1e-5, "oracle_precision", path, "float32 action Oracle error <1e-5", oracle["error"])
        self.scalar(oracle.get("direct_float64_series_error"), 0., path / "direct_float64_series_error")
        self.check(oracle.get("direct_float64_bitwise_equal") is True, "direct_oracle_exact", path, True, oracle.get("direct_float64_bitwise_equal"))
        with np.load(path.parent / "rollout.npz", allow_pickle=False) as archive:
            controls = archive["physical_controls"].copy()
        self.rollout(path.parent / "rollout.npz", {**oracle, "tracking": self.tracking(controls, self.target_controls)})
        identity_path = path.parent / "target_identity.json"
        identity = self.read_json(identity_path)
        records = identity["checks"]
        expected = {(environment, seed, episode) for environment in ("training", "evaluation") for seed in (0, 1, 2, 10000) for episode in range(3)}
        actual = {(record.get("environment"), record.get("seed"), record.get("episode")) for record in records}
        self.check(identity.get("passed") is True and len(records) == 24 and actual == expected,
                   "target_identity_set", identity_path, "24 training/evaluation seed/reset checks", len(records))
        for record in records:
            self.check(record.get("target_bitwise_equal") is True and record.get("controls_bitwise_equal") is True and
                       record.get("phase") == 0 and record.get("index") == 0 and record.get("previous_zero") is True,
                       "target_and_reset_identity", identity_path, "identical target and reset state", record)


def audit(output):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / "results").resolve()) or not (output / "manifest.json").is_file():
        raise ValueError("Audit output must be a prepared new Phase 0 experiment below Deeplearning/results")
    manifest = json.loads((output / "manifest.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("workflow") != "issue14_phase0_fixed_target":
        raise ValueError("Only an Issue #14 experiment may receive audit.json")
    result = Auditor(output).run()
    destination = output/"reward_ablation/control_audit.json" if (output/"reward_ablation").is_dir() else output/"audit.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    if not result["passed"]:
        raise RuntimeError(f"Issue #14 audit failed: {len(result['errors'])} errors; see {output/'audit.json'}")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({key: result[key] for key in ("passed", "checks", "counts", "wall_s")}))


if __name__ == "__main__":
    main()
