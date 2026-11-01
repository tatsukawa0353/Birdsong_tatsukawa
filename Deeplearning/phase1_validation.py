"""Staged, synthetic-only Phase 1 validation. Historical experiments are immutable.

Run prepare, inspect diagnostics_review.json, then run A, B, C in that order.
Each optimizer searches the same physical trajectory space as its RL comparator.
No module in this workflow loads recordings or private_runs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
from pathlib import Path
import time

import numpy as np
from scipy.optimize import differential_evolution

from .acoustics import distance
from .model import BirdModel, DEFAULT_CONTROLS, DT, HIGH, LOW, ROOT
from .model_env import AudioDecimator, ModelConfig, ModelEnv
from .objectives import BalancedObjective


CONFIG = ROOT / "configs/phase1_validation.json"
DEFAULT_OUTPUT = ROOT / "results/phase1_validation_20261031"
TARGET = np.array([4.1e7, 3.5e7, 3.4e6, 4.9e4, 4.9e4])
RATE, DURATION, CONTROL_S = 50000, .35, .01


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_files(output):
    """Record every preexisting synthetic result and unchanged physical/loss code."""
    paths = [p for p in (ROOT / "results").rglob("*")
             if p.is_file() and output not in p.parents]
    paths += [ROOT / name for name in ("model.py", "model_env.py", "acoustics.py",
                                      "objectives.py", "plotting.py")]
    paths += list((ROOT / "native").glob("*.cpp"))
    paths += list(ROOT.parent.glob("BirdsongModel*"))
    return {str(p.relative_to(ROOT.parent)): digest(p) for p in paths if p.is_file()}


def source_hashes():
    names = ("phase1_validation.py", "phase1_validation_env.py",
             "phase1_validation_report.py", "configs/phase1_validation.json")
    return {name: digest(ROOT / name) for name in names}


def render(model, controls):
    model.reset()
    decimator = AudioDecimator()
    audio = []
    for c in controls:
        model.set_controls(c)
        audio.append(decimator.process(model.advance(CONTROL_S)))
    return np.concatenate(audio)


def scores(target, audio, controls, indices, *, failed=False, objective=None):
    if failed or len(audio) != len(target):
        return dict(acoustic_error=1e6, balanced_error=1e6,
                    parameter_nrmse=1e6, failed=True)
    acoustic = distance(target, audio, RATE)
    objective = objective or BalancedObjective(target, RATE, scale=max(abs(target).max(), 1e-8))
    balanced = objective.score(audio)
    err = (np.asarray(controls)[:, indices]-TARGET[indices])/(HIGH[indices]-LOW[indices])
    return dict(acoustic_error=acoustic["total"], balanced_error=balanced["total"],
                parameter_nrmse=float(np.sqrt(np.mean(err**2))),
                acoustic_components=acoustic, balanced_components=balanced, failed=False)


def episode(env, actions=None, policy=None):
    obs, _ = env.reset()
    rewards = []
    failed = False
    for k in range(env.horizon):
        action = policy(obs) if policy else np.asarray(actions)[k]
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        if done or truncated:
            failed = bool(info.get("numerical_failure", False))
            break
    audio = np.concatenate(env.audio)
    controls = np.asarray(env.applied_control_history)
    metrics = scores(env.target_audio, audio, controls, env.indices, failed=failed)
    metrics["negative_return"] = -float(sum(rewards))
    return metrics, audio.copy(), controls.copy(), np.asarray(env.actions).copy()


def account(env, wall_s):
    return dict(decision_steps=env.decision_steps, episodes=env.completed_episodes,
                integration_s=env.integrated_s, integration_completed_s=env.integrated_completed_s,
                native_steps=round(env.integrated_s/DT), native_blocks=env.native_blocks,
                wall_s=float(wall_s), failures=env.failures)


def load_ready(output):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["source_hashes"] != source_hashes():
        raise RuntimeError("Experiment source/config changed: use a new output directory")
    gate = json.loads((output / "diagnostics_review.json").read_text())
    if not gate["passed"]:
        raise RuntimeError("Oracle/landscape gate has not passed")
    with np.load(output / "diagnostics/target.npz") as data:
        target = data["target"].copy()
    return manifest, target


def prepare(output, config):
    from .phase1_validation_env import ValidationEnv
    from .phase1_validation_report import plot_landscapes, save_rollout
    if config["gamma"] != 1.0:
        raise ValueError("Prefix-potential validation requires gamma=1")
    if len(set(config["seeds"])) < 3:
        raise ValueError("At least three distinct seeds are required")
    if output.exists():
        raise FileExistsError("prepare requires a new results directory")
    output.mkdir(parents=True)
    out = output / "diagnostics"
    out.mkdir()
    start = time.perf_counter()
    protected = protected_files(output)
    write_json(output / "protected_before.json", protected)
    target_start = time.perf_counter()
    env = ModelEnv()
    target = env.target_audio.copy()
    controls = env.target_controls.copy()
    env.close()
    target_wall = time.perf_counter()-target_start
    np.savez_compressed(out / "target.npz", target=target, controls=controls, rate=RATE)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    settings=config, source_hashes=source_hashes(),
                    condition="A", target="constant model-generated, never real recordings",
                    duration_s=DURATION, control_interval_s=CONTROL_S, sample_rate=RATE,
                    native_dt=DT, target_controls=TARGET.tolist(),
                    fixed_controls=TARGET.tolist(), physical_low=LOW.tolist(), physical_high=HIGH.tolist(),
                    initial_state=[0, 0, 0, 0], delay_history="all zeros", decimator_state="all zeros",
                    target_generation=dict(integration_s=DURATION, native_steps=round(DURATION/DT), wall_s=target_wall),
                    reward="whole_acoustic_prefix_potential_gamma1; NOT baseline_context or balanced_aligned",
                    reward_identity="-sum(r_t) = full acoustic error; potential=(n/N)*distance(prefix)",
                    primary="acoustics.distance over the complete 0.35 s", balanced="audit only, not training/selection",
                    optimizer_space=dict(A="one constant epsilon_l", B="35 independent epsilon_l blocks",
                                         C="35 independent (epsilon_l,ps) blocks"),
                    budget_unit="training episodes of 0.35 s, equal within each task",
                    evaluation_schedule="initial plus 8 checkpoints; evaluation integration excluded from training budget")
    manifest["runtime"] = dict(python=platform.python_version(), platform=platform.platform(),
                                package_versions={name: importlib.metadata.version(name) for name in
                                                  ("numpy", "scipy", "torch", "stable-baselines3", "gymnasium")},
                                torch_threads=1, comparison_workers=3,
                                requirements_lock_sha256=digest(ROOT / "requirements-lock.txt"))
    manifest["reproduction_commands"] = [
        "MPLCONFIGDIR=/tmp/birdsong-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
        "Deeplearning/.venv/bin/python -m Deeplearning.phase1_validation "+stage
        for stage in ("prepare", "baselines", "compare --task A", "compare --task B", "compare --task C", "report")]
    write_json(output / "manifest.json", manifest)
    records = []
    # Show the original fixed-pressure mismatch using its unmodified historical environment.
    old = ModelEnv(ModelConfig(control_indices=(0,)), external_target=target)
    old.reset()
    for _ in range(old.horizon):
        old.step(old.encode(TARGET))
    old_metrics = scores(target, np.concatenate(old.audio), old.applied_control_history, [0])
    records.append(dict(name="historical_1d_decode_mismatch", fixed_ps=3e6, **old_metrics))
    old.close()
    for task in ("A", "B", "C"):
        for dtype in (np.float64, np.float32):
            e = ValidationEnv(task, target_audio=target)
            actions = np.tile(e.encode(TARGET, dtype=dtype), (e.horizon, 1))
            oracle_start = time.perf_counter()
            result, audio, applied, actual = episode(e, actions)
            result.update(name=f"{task}_oracle_{np.dtype(dtype).name}", task=task,
                          control_dtype=np.dtype(dtype).name,
                          max_waveform_abs_error=float(np.max(abs(audio-target))),
                          **account(e, time.perf_counter()-oracle_start))
            records.append(result)
            if dtype == np.float64:
                save_rollout(out / f"oracle_{task}", target, audio, applied, actual, RATE)
            e.close()
    write_json(out / "oracles.json", records)
    objective = BalancedObjective(target, RATE, scale=max(abs(target).max(), 1e-8))
    map_summaries = {}
    # Exact known controls are diagnostic probes, never supplied to an optimizer.
    epsilon1 = np.unique(np.r_[np.linspace(LOW[0], HIGH[0], config["landscape_epsilon_points_1d"]), TARGET[0]])
    epsilon2 = np.unique(np.r_[np.linspace(LOW[0], HIGH[0], config["landscape_epsilon_points_2d"]), TARGET[0]])
    pressure = np.unique(np.r_[np.linspace(LOW[2], HIGH[2], config["landscape_pressure_points_2d"]), TARGET[2]])
    with BirdModel("A") as model:
        for name, candidates in (("1d", [(x, TARGET[2]) for x in epsilon1]),
                                 ("2d", [(x, y) for y in pressure for x in epsilon2])):
            wall = time.perf_counter()
            rows = []
            for x, y in candidates:
                tick = time.perf_counter()
                c = np.tile(TARGET, (35, 1))
                c[:, 0], c[:, 2] = x, y
                audio = render(model, c)
                result = scores(target, audio, c, [0, 2], objective=objective)
                rows.append(dict(epsilon_l=float(x), ps=float(y), acoustic_error=result["acoustic_error"],
                                 balanced_error=result["balanced_error"], wall_s=time.perf_counter()-tick))
            with (out / f"landscape_{name}.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            map_summaries[name] = dict(candidates=len(rows), integration_s=DURATION*len(rows),
                                       native_steps=round(DURATION*len(rows)/DT), wall_s=time.perf_counter()-wall,
                                       minimum_acoustic_error=min(r["acoustic_error"] for r in rows),
                                       maximum_acoustic_error=max(r["acoustic_error"] for r in rows),
                                       below_008=sum(r["acoustic_error"] <= .08 for r in rows),
                                       finite=bool(all(np.isfinite(r["acoustic_error"]) and np.isfinite(r["balanced_error"]) for r in rows)))
            print(f"landscape {name}: {map_summaries[name]}", flush=True)
    plot_landscapes(output)
    checks = {r["name"]: r["acoustic_error"] <= config["oracle_tolerance_"+r["control_dtype"]]
              for r in records if "control_dtype" in r}
    checks.update({name+"_finite": value["finite"] for name, value in map_summaries.items()})
    gate = dict(passed=all(checks.values()), checks=checks, maps=map_summaries,
                reviewed_utc=datetime.now(timezone.utc).isoformat(),
                diagnosis="Historical fixed p_s mismatch corrected only in new environment. Known optimum reachable. Coarse map suffices for reachability; no adaptive refinement or map-based optimizer initialization.",
                prepare_wall_s=time.perf_counter()-start,
                oracle_integration_s=7*DURATION,
                decision="Proceed to A, then B, then C with predeclared budgets; failures remain reportable.")
    write_json(output / "diagnostics_review.json", gate)
    if not gate["passed"]:
        raise RuntimeError("Oracle/map consistency gate failed")
    print(json.dumps(gate, ensure_ascii=False), flush=True)


def run_one(output, task, method, seed):
    from .phase1_validation_env import ValidationEnv
    from .phase1_validation_report import save_rollout
    manifest, target = load_ready(output)
    config = manifest["settings"]
    run_dir = output / "runs" / task / method / f"seed_{seed}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {run_dir}")
    run_dir.mkdir(parents=True)
    start = time.perf_counter()
    train_env, eval_env = ValidationEnv(task, target_audio=target), ValidationEnv(task, target_audio=target)
    spec = config["tasks"][task]
    budget, interval = spec["episodes"], spec["eval_every_episodes"]
    curve = []
    evaluation_wall = 0.0
    snapshots = {}

    def evaluate(n, actions=None, policy=None):
        nonlocal evaluation_wall
        tick = time.perf_counter()
        result, audio, controls, actual = episode(eval_env, actions=actions, policy=policy)
        curve.append(dict(episodes=n, decision_steps=train_env.decision_steps,
                          integration_s=train_env.integrated_s, **result))
        snapshots["final"] = (audio, controls, actual)
        if n == 0:
            snapshots["initial"] = (audio, controls, actual)
        evaluation_wall += time.perf_counter()-tick
        return result

    dimensions = train_env.horizon*len(train_env.indices)
    gradient_updates = 0
    optimizer_nfev = 0
    if method in ("TD3", "SAC"):
        import torch
        from stable_baselines3 import TD3, SAC
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.noise import NormalActionNoise
        torch.set_num_threads(1)
        kwargs = dict(policy="MlpPolicy", env=train_env, seed=seed, device="cpu", verbose=0,
                      policy_kwargs=dict(net_arch=config["net_arch"]), learning_rate=config["learning_rate"],
                      gamma=config["gamma"], batch_size=config["batch_size"], buffer_size=50000,
                      learning_starts=config["warmup_episodes"]*train_env.horizon,
                      train_freq=1, gradient_steps=1)
        if method == "TD3":
            kwargs["action_noise"] = NormalActionNoise(np.zeros(len(train_env.indices)),
                                                       config["td3_noise_std"]*np.ones(len(train_env.indices)))
            agent = TD3(**kwargs)
        else:
            agent = SAC(**kwargs, ent_coef=config["sac_ent_coef"])
        policy = lambda obs: agent.predict(obs, deterministic=True)[0]
        evaluate(0, policy=policy)

        class Checkpoint(BaseCallback):
            def _on_step(self):
                return True

            def _on_rollout_end(self):
                # Invoked before the next gradient update; all 35 transitions in
                # this episode are available. Final evaluation follows learn().
                n = train_env.completed_episodes
                if (n and n < budget and n % interval == 0
                        and curve[-1]["episodes"] != n):
                    evaluate(n, policy=policy)

        agent.learn(total_timesteps=budget*train_env.horizon, callback=Checkpoint())
        evaluate(budget, policy=policy)
        gradient_updates = int(agent._n_updates)
        agent.save(run_dir / "agent.zip")
    else:
        rng = np.random.default_rng(seed)
        initial = rng.uniform(-1, 1, size=(train_env.horizon, len(train_env.indices)))
        best = initial.copy()
        initial_metrics = evaluate(0, actions=best)
        best_error = initial_metrics["acoustic_error"]

        def candidate(flat):
            nonlocal best, best_error, optimizer_nfev
            actions = np.asarray(flat).reshape(train_env.horizon, len(train_env.indices))
            result, _, _, _ = episode(train_env, actions)
            optimizer_nfev += 1
            if result["acoustic_error"] < best_error:
                best_error, best = result["acoustic_error"], actions.copy()
            if optimizer_nfev % interval == 0:
                evaluate(optimizer_nfev, actions=best)
            return result["acoustic_error"]

        if method == "random_search":
            # Initial evaluation is also the first counted training candidate,
            # as in DE's explicit population. It grants no extra search trial.
            candidate(initial.ravel())
            for _ in range(budget-1):
                candidate(rng.uniform(-1, 1, size=dimensions))
        elif method == "DE":
            pop = config["de_population"]
            if budget % pop:
                raise ValueError("DE budget must be divisible by explicit population size")
            population = rng.uniform(-1, 1, size=(pop, dimensions))
            population[0] = initial.ravel()
            optimized = differential_evolution(candidate, [(-1, 1)]*dimensions, init=population,
                                               maxiter=budget//pop-1, polish=False, tol=0, atol=0,
                                               mutation=tuple(config["de_mutation"]),
                                               recombination=config["de_recombination"], rng=rng)
            # A converged population can legitimately stop before the cap.
            if curve[-1]["episodes"] != optimizer_nfev:
                evaluate(optimizer_nfev, actions=best)
            np.save(run_dir / "optimizer_best_actions.npy", best)
            if optimized.nfev != optimizer_nfev:
                raise RuntimeError("DE evaluation accounting mismatch")
        else:
            raise ValueError(method)
    elapsed = time.perf_counter()-start
    # Save both initial and final using identical original-amplitude output paths.
    for name, values in snapshots.items():
        destination = run_dir if name == "final" else run_dir / name
        save_rollout(destination, target, *values, RATE)
    summary = dict(task=task, method=method, seed=seed, optimizer_dimension=dimensions,
                   initial_metrics={k: v for k, v in curve[0].items()
                                    if k not in ("episodes", "decision_steps", "integration_s")},
                   final_metrics={k: v for k, v in curve[-1].items()
                                  if k not in ("episodes", "decision_steps", "integration_s")},
                   training=account(train_env, elapsed-evaluation_wall),
                   evaluation=account(eval_env, evaluation_wall), elapsed_s=elapsed,
                   artifact_wall_s=time.perf_counter()-start-elapsed,
                   requested_training_episodes=budget, optimizer_nfev=optimizer_nfev,
                   gradient_updates=gradient_updates, reward_profile=manifest["reward"],
                   optimizer_objective="acoustic_error", balanced_role="independent audit",
                   returned_solution="final deterministic actor" if method in ("TD3", "SAC") else "best training candidate; initial candidate counted in training",
                   source_hashes=manifest["source_hashes"])
    write_json(run_dir / "curve.json", curve)
    write_json(run_dir / "summary.json", summary)
    train_env.close()
    eval_env.close()
    print(json.dumps(dict(task=task, method=method, seed=seed,
                          error=summary["final_metrics"]["acoustic_error"], elapsed_s=elapsed)), flush=True)


def baselines(output):
    from .phase1_validation_env import ValidationEnv
    from .phase1_validation_report import save_rollout
    manifest, target = load_ready(output)
    root = output / "baselines"
    if root.exists():
        raise FileExistsError("Refusing to overwrite baselines")
    root.mkdir()
    results = []
    for task in ("A", "B", "C"):
        for seed in manifest["settings"]["seeds"]:
            for name in ("neutral", "random_control"):
                env = ValidationEnv(task, target_audio=target)
                rng = np.random.default_rng(seed+10000)
                actions = (np.zeros((env.horizon, len(env.indices))) if name == "neutral" else
                           rng.uniform(-1, 1, size=(env.horizon, len(env.indices))))
                tick = time.perf_counter()
                result, audio, controls, actual = episode(env, actions)
                results.append(dict(task=task, seed=seed, name=name, **result,
                                    compute=account(env, time.perf_counter()-tick)))
                save_rollout(root / task / name / f"seed_{seed}", target, audio, controls, actual, RATE)
                env.close()
    write_json(root / "summary.json", results)


def compare(output, task, workers):
    manifest, _ = load_ready(output)
    settings = manifest["settings"]
    for earlier in "ABC"[:"ABC".index(task)]:
        required = [output / "runs" / earlier / method / f"seed_{seed}" / "summary.json"
                    for method in settings["methods"] for seed in settings["seeds"]]
        if not all(p.exists() for p in required):
            raise RuntimeError(f"Complete task {earlier} before task {task}")
    jobs = []
    for method in settings["methods"]:
        for seed in settings["seeds"]:
            summary = output / "runs" / task / method / f"seed_{seed}" / "summary.json"
            if summary.exists():
                saved = json.loads(summary.read_text())
                if saved["source_hashes"] != manifest["source_hashes"]:
                    raise RuntimeError("Cannot reuse results from different code")
            else:
                jobs.append((method, seed))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_one, output, task, method, seed) for method, seed in jobs]
        for future in futures:
            future.result()


def verify_protected(output):
    before = json.loads((output / "protected_before.json").read_text())
    after = protected_files(output)
    changed = [name for name in before.keys() | after.keys() if before.get(name) != after.get(name)]
    write_json(output / "preservation_check.json", dict(passed=not changed, files_checked=len(before), changed=changed))
    if changed:
        raise RuntimeError("Pre-existing physical code or results changed")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["prepare", "baselines", "run", "compare", "report"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--task", choices=["A", "B", "C"])
    parser.add_argument("--method", choices=["TD3", "SAC", "random_search", "DE"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Synthetic-only workflow writes below Deeplearning/results")
    if args.stage == "prepare":
        prepare(output, json.loads(args.config.read_text()))
    elif args.stage == "baselines":
        baselines(output)
    elif args.stage == "run":
        if args.task is None or args.method is None or args.seed is None:
            parser.error("run requires --task, --method, --seed")
        run_one(output, args.task, args.method, args.seed)
    elif args.stage == "compare":
        if args.task is None or not 1 <= args.workers <= 3:
            parser.error("compare requires --task and 1..3 workers")
        compare(output, args.task, args.workers)
    else:
        from .phase1_validation_report import report
        load_ready(output)
        verify_protected(output)
        report(output)


if __name__ == "__main__":
    main()
