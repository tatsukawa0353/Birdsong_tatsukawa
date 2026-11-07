"""Issue #13: isolated constant-control reproduction, never recording imitation.

Reuse the original physics, objective, policy settings and pre-update evaluation
timing. Only the evaluation density and diagnostic retention change in 1D.
The 2D condition adds pressure while retaining one decision per 0.35 s episode.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.io import wavfile
from scipy.optimize import differential_evolution

from .issue13_env import ConstantControlEnv
from .model import DT, HIGH, LOW, ROOT, LIBRARY
from .phase1_validation import account, episode
from .phase1_validation_env import DURATION, RATE, TARGET_CONTROLS


CONFIG = ROOT / "configs/issue13_reproduction.json"
DEFAULT_OUTPUT = ROOT / "results/phase1_issue13_20261107"
WORKFLOW = "issue13_constant_control_reproduction"
INDICES = {"constant_1d": (0,), "constant_2d": (0, 2)}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    names = ("issue13.py", "issue13_env.py", "configs/issue13_reproduction.json",
             "phase1_validation.py", "phase1_validation_env.py", "model.py", "model_env.py",
             "acoustics.py", "objectives.py", "native/model.cpp", "requirements-lock.txt")
    return {name: digest(ROOT / name) for name in names}


def protected_files(output):
    excluded = {".venv", "venv", "build", "__pycache__", ".pytest_cache", "private_runs", "sample_song"}
    files = {}
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in excluded and (Path(base) / d).resolve() != output]
        for name in names:
            p = Path(base) / name
            if name.startswith("issue13") or name == "test_issue13.py" or name == "ISSUE13_REPORT.md":
                continue
            if p.is_file():
                files[str(p.relative_to(ROOT.parent))] = digest(p)
    # Preserve the original research model sources as well as the active bridge.
    for p in ROOT.parent.glob("BirdsongModel*"):
        if p.is_file():
            files[str(p.relative_to(ROOT.parent))] = digest(p)
    return files


def verify_protected(output):
    before = json.loads((output / "protected_before.json").read_text())
    after = protected_files(output)
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    result = dict(passed=not changed, files_checked=len(before), changed=changed)
    write_json(output / "preservation_check.json", result)
    if changed:
        raise RuntimeError(f"Protected files changed: {changed}")
    return result


def load_ready(output):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("workflow") != WORKFLOW or manifest["source_hashes"] != source_hashes():
        raise RuntimeError("Computation source/settings changed; use a new experiment directory")
    if not json.loads((output / "diagnostics/oracle_gate.json").read_text())["passed"]:
        raise RuntimeError("Oracle gate did not pass")
    if digest(LIBRARY) != manifest["native_library_sha256"]:
        raise RuntimeError("Native library changed")
    with np.load(output / "diagnostics/target.npz", allow_pickle=False) as data:
        target = data["target"].copy()
    return manifest, target


def prepare(output, settings):
    if output.exists():
        raise FileExistsError("Use a new output directory; old experiments are immutable")
    if (settings["seeds"] != [0, 1, 2] or settings["episodes"] != 256 or
            settings["eval_every_episodes"] != 1 or settings["gamma"] != 1.0):
        raise ValueError("The reproduction uses seeds 0/1/2, 256 trials, evaluation every episode and gamma=1")
    reference = ROOT.parent / settings["historical_reference_directory"]
    previous = json.loads((reference / "manifest.json").read_text())
    keys = ("warmup_episodes", "learning_rate", "batch_size", "net_arch", "gamma", "sac_ent_coef",
            "td3_noise_std", "de_population", "de_mutation", "de_recombination")
    if any(settings[key] != previous["settings"][key] for key in keys):
        raise ValueError("Reproduction settings differ from the historical main comparison")
    output.mkdir(parents=True)
    diagnostics = output / "diagnostics"
    diagnostics.mkdir()
    write_json(output / "protected_before.json", protected_files(output))
    original = reference / "diagnostics/target.npz"
    shutil.copy2(original, diagnostics / "target.npz")
    with np.load(original, allow_pickle=False) as data:
        target = data["target"].copy()
    manifest = dict(workflow=WORKFLOW, issue_url="https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/13",
                    experiment_id=settings["experiment_id"], created_utc=datetime.now(timezone.utc).isoformat(),
                    client_date_jst="2026-11-07", settings=settings, source_hashes=source_hashes(),
                    historical_reference_directory=str(reference.resolve()), condition="A",
                    target="same constant model-generated waveform as historical Task A; no recordings",
                    target_controls=list(TARGET_CONTROLS), fixed_controls=list(TARGET_CONTROLS),
                    physical_low=LOW.tolist(), physical_high=HIGH.tolist(), duration_s=DURATION,
                    sample_rate=RATE, native_dt=DT, control_interval_s=.01,
                    initial_state=[0, 0, 0, 0], delay_history="all zeros", decimator_state="all zeros",
                    source_target_sha256=digest(original), copied_target_sha256=digest(diagnostics / "target.npz"),
                    target_generation=dict(episodes=0, integration_s=0, note="copied saved target, no new generation"),
                    reward=previous["reward"], reward_identity="-sum(r_t) = full acoustic error; gamma=1",
                    primary="full 0.35 s acoustics.distance; balanced is audit only",
                    optimizer_space={name: list(indices) for name, indices in INDICES.items()},
                    budget_unit="256 training sound generations per method/condition/seed",
                    evaluation_schedule="RL: 0 plus each episode 1..256; direct: objective/best-so-far for every candidate",
                    evaluation_timing="same historical pre-gradient rollout-end at 1..255; final 256 post-final-update",
                    differences_from_historical=["evaluation interval 32 to 1", "retain evaluation actions and best checkpoint/audio",
                                                 "new constant two-variable condition; no hyperparameter improvement"],
                    native_library_sha256=digest(LIBRARY),
                    package_versions={name: importlib.metadata.version(name) for name in
                                      ("numpy", "scipy", "torch", "stable-baselines3", "gymnasium")})
    write_json(output / "manifest.json", manifest)
    write_json(output / "config.json", settings)
    for name in manifest["source_hashes"]:
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    oracles = []
    for name, indices in INDICES.items():
        for dtype in (np.float64, np.float32):
            env = ConstantControlEnv(indices, target_audio=target)
            try:
                tick = time.perf_counter()
                result, audio, controls, actions = episode(env, actions=[env.encode(TARGET_CONTROLS, dtype=dtype)])
                oracles.append(dict(condition=name, dtype=np.dtype(dtype).name, **result,
                                    max_waveform_abs_error=float(np.max(np.abs(audio-target))),
                                    bitwise_equal=bool(np.array_equal(audio, target)),
                                    compute=account(env, time.perf_counter()-tick)))
                if dtype == np.float64:
                    save_snapshot(diagnostics / f"oracle_{name}", target, (audio, controls, actions), plots=False)
            finally:
                env.close()
    write_json(diagnostics / "oracles.json", oracles)
    passed = all(r["bitwise_equal"] and r["acoustic_error"] == 0 for r in oracles if r["dtype"] == "float64")
    passed &= all(r["acoustic_error"] < 1e-5 for r in oracles)
    write_json(diagnostics / "oracle_gate.json", dict(passed=bool(passed), oracle_count=len(oracles)))
    if not passed:
        raise RuntimeError("Oracle consistency failed; do not train")
    print(json.dumps(dict(stage="prepare", oracle_gate="PASS", output=str(output))), flush=True)


def save_snapshot(destination, target, values, *, plots):
    audio, controls, actions = values
    if plots:
        from .phase1_validation_report import save_rollout
        save_rollout(destination, target, audio, controls, actions, RATE)
        return
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "rollout.npz", target=target, audio=audio,
                        controls=controls, actions=actions, rate=np.asarray(RATE))
    wavfile.write(destination / "target_raw.wav", RATE, target.astype(np.float32))
    wavfile.write(destination / "generated_raw.wav", RATE, audio.astype(np.float32))
    write_json(destination / "audio_format.json", dict(npz_dtype="float64", wav_format="IEEE float32",
               sample_rate=RATE, independent_gain_normalization=False,
               target_peak=float(np.max(np.abs(target))), generated_peak=float(np.max(np.abs(audio))),
               units="model output; no calibrated sound-pressure conversion", sap_evaluated=False))


def run_one(output, condition, method, seed):
    manifest, target = load_ready(output)
    config = manifest["settings"]
    run_dir = output / "runs" / condition / method / f"seed_{seed}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", dict(condition=condition, method=method, seed=seed,
               settings=config, control_indices=list(INDICES[condition]), source_hashes=manifest["source_hashes"]))
    start = time.perf_counter()
    train_env = ConstantControlEnv(INDICES[condition], target_audio=target)
    eval_env = ConstantControlEnv(INDICES[condition], target_audio=target)
    budget, curve = config["episodes"], []
    snapshots, best_record = {}, None
    evaluation_wall, artifact_wall, gradient_updates, optimizer_nfev = 0., 0., 0, 0
    agent = None

    def append_record(n, metrics, values, phase, *, extra=None):
        nonlocal best_record, artifact_wall
        audio, controls, actions = values
        record = dict(episodes=n, decision_steps=train_env.decision_steps,
                      integration_s=train_env.integrated_s, **metrics,
                      actions=actions.tolist(), physical_controls=controls.tolist(),
                      gradient_updates=int(agent._n_updates) if agent is not None else 0,
                      evaluation_phase=phase)
        if extra:
            record.update(extra)
        record["best_observed_error"] = min(metrics["acoustic_error"],
                                             best_record["acoustic_error"] if best_record is not None else float("inf"))
        curve.append(record)
        snapshots["final"] = (audio.copy(), controls.copy(), actions.copy())
        if n == 0:
            snapshots["initial"] = snapshots["final"]
        if best_record is None or metrics["acoustic_error"] < best_record["acoustic_error"]:
            best_record = record.copy()
            snapshots["best_observed"] = snapshots["final"]
            if agent is not None:
                tick = time.perf_counter()
                (run_dir / "best_observed").mkdir(exist_ok=True)
                agent.save(run_dir / "best_observed/agent.zip")
                artifact_wall += time.perf_counter()-tick
        return record

    def evaluate(n, *, actions=None, policy=None, phase):
        nonlocal evaluation_wall
        tick = time.perf_counter()
        metrics, audio, controls, actual = episode(eval_env, actions=actions, policy=policy)
        evaluation_wall += time.perf_counter()-tick
        return append_record(n, metrics, (audio, controls, actual), phase)

    print(json.dumps(dict(stage="start", condition=condition, method=method, seed=seed)), flush=True)
    try:
        if method in ("TD3", "SAC"):
            import torch
            from stable_baselines3 import TD3, SAC
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.noise import NormalActionNoise
            torch.set_num_threads(1)
            kwargs = dict(policy="MlpPolicy", env=train_env, seed=seed, device="cpu", verbose=0,
                          policy_kwargs=dict(net_arch=config["net_arch"]), learning_rate=config["learning_rate"],
                          gamma=config["gamma"], batch_size=config["batch_size"], buffer_size=50000,
                          learning_starts=config["warmup_episodes"], train_freq=1, gradient_steps=1)
            if method == "TD3":
                kwargs["action_noise"] = NormalActionNoise(np.zeros(len(train_env.indices)),
                                                          config["td3_noise_std"]*np.ones(len(train_env.indices)))
                agent = TD3(**kwargs)
            else:
                agent = SAC(**kwargs, ent_coef=config["sac_ent_coef"])
            policy = lambda obs: agent.predict(obs, deterministic=True)[0]
            evaluate(0, policy=policy, phase="initial")

            class DenseCheckpoint(BaseCallback):
                def _on_step(self):
                    return True

                def _on_rollout_end(self):
                    n = train_env.completed_episodes
                    if 0 < n < budget:
                        evaluate(n, policy=policy, phase="pre_gradient_update")
                        if n % 64 == 0:
                            print(json.dumps(dict(stage="progress", condition=condition, method=method,
                                                  seed=seed, trials=n, error=curve[-1]["acoustic_error"])), flush=True)

            tick, eval_before, artifact_before = time.perf_counter(), evaluation_wall, artifact_wall
            agent.learn(total_timesteps=budget, callback=DenseCheckpoint())
            training_wall = time.perf_counter()-tick-(evaluation_wall-eval_before)-(artifact_wall-artifact_before)
            evaluate(budget, policy=policy, phase="post_final_update")
            gradient_updates = int(agent._n_updates)
        else:
            rng = np.random.default_rng(seed)
            initial_actions = rng.uniform(-1, 1, size=(1, len(train_env.indices)))
            initial_record = evaluate(0, actions=initial_actions, phase="initial")
            best_actions, best_error = initial_actions.copy(), initial_record["acoustic_error"]
            best_metrics = {key: initial_record[key] for key in ("acoustic_error", "balanced_error", "parameter_nrmse",
                            "acoustic_components", "balanced_components", "negative_return", "failed")}
            best_values = snapshots["initial"]

            def candidate(flat):
                nonlocal best_actions, best_error, best_metrics, best_values, optimizer_nfev
                actions = np.asarray(flat).reshape(1, len(train_env.indices))
                metrics, audio, controls, actual = episode(train_env, actions=actions)
                optimizer_nfev += 1
                if metrics["acoustic_error"] < best_error:
                    best_error, best_actions = metrics["acoustic_error"], actual.copy()
                    best_metrics, best_values = metrics, (audio, controls, actual)
                append_record(optimizer_nfev, best_metrics, best_values, "candidate_best_so_far",
                              extra=dict(candidate_error=metrics["acoustic_error"],
                                         candidate_actions=actual.tolist(), candidate_physical_controls=controls.tolist(),
                                         candidate_components=metrics["acoustic_components"], candidate_failed=metrics["failed"]))
                return metrics["acoustic_error"]

            tick = time.perf_counter()
            if method == "random_search":
                candidate(initial_actions.ravel())
                for _ in range(budget-1):
                    candidate(rng.uniform(-1, 1, size=len(train_env.indices)))
            elif method == "DE":
                pop = config["de_population"]
                population = rng.uniform(-1, 1, size=(pop, len(train_env.indices)))
                population[0] = initial_actions.ravel()
                result = differential_evolution(candidate, [(-1, 1)]*len(train_env.indices), init=population,
                            maxiter=budget//pop-1, polish=False, tol=0, atol=0,
                            mutation=tuple(config["de_mutation"]), recombination=config["de_recombination"], rng=rng)
                if result.nfev != budget or optimizer_nfev != budget:
                    raise RuntimeError("DE stopped before the specified 256-candidate budget")
            else:
                raise ValueError(method)
            training_wall = time.perf_counter()-tick
            tick = time.perf_counter()
            final_metrics, final_audio, final_controls, final_actions = episode(eval_env, actions=best_actions)
            evaluation_wall += time.perf_counter()-tick
            if final_metrics["acoustic_error"] != curve[-1]["acoustic_error"]:
                raise RuntimeError("Final independent replay does not match the selected objective")
            snapshots["final"] = (final_audio, final_controls, final_actions)
            np.save(run_dir / "optimizer_best_actions.npy", best_actions)

        tick = time.perf_counter()
        if agent is not None:
            agent.save(run_dir / "agent.zip")
        for stage, values in snapshots.items():
            destination = run_dir if stage == "final" else run_dir / stage
            save_snapshot(destination, target, values, plots=stage == "final")
        artifact_wall += time.perf_counter()-tick
        write_json(run_dir / "curve.json", curve)
        summary = dict(condition=condition, method=method, seed=seed,
                       optimizer_dimension=len(train_env.indices), initial=curve[0], final=curve[-1],
                       best_observed=best_record, best_observed_episode=best_record["episodes"],
                       initial_metrics=curve[0], final_metrics=curve[-1], best_observed_metrics=best_record,
                       training=account(train_env, training_wall), evaluation=account(eval_env, evaluation_wall),
                       requested_training_episodes=budget, optimizer_nfev=optimizer_nfev, gradient_updates=gradient_updates,
                       elapsed_s=time.perf_counter()-start, artifact_wall_s=artifact_wall,
                       returned_solution="final deterministic actor" if agent is not None else "best training candidate; initial counted in training",
                       reward_profile=manifest["reward"], optimizer_objective="acoustic_error", balanced_role="audit only",
                       best_checkpoint="best_observed/agent.zip" if agent is not None else None,
                       source_hashes=manifest["source_hashes"])
        write_json(run_dir / "summary.json", summary)
        print(json.dumps(dict(stage="complete", condition=condition, method=method, seed=seed,
                              error=summary["final"]["acoustic_error"], best=best_record["acoustic_error"],
                              elapsed_s=summary["elapsed_s"])), flush=True)
        return summary
    finally:
        train_env.close()
        eval_env.close()


def compare(output, condition, workers):
    manifest, _ = load_ready(output)
    jobs = []
    for method in manifest["settings"]["methods"]:
        for seed in manifest["settings"]["seeds"]:
            run_dir = output / "runs" / condition / method / f"seed_{seed}"
            if (run_dir / "summary.json").exists():
                summary = json.loads((run_dir / "summary.json").read_text())
                if summary["source_hashes"] != manifest["source_hashes"]:
                    raise RuntimeError("Completed run source mismatch")
            else:
                jobs.append((method, seed))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_one, output, condition, method, seed) for method, seed in jobs]
        for future in futures:
            future.result()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["prepare", "compare", "run", "report", "all"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--condition", choices=list(INDICES))
    parser.add_argument("--method", choices=["TD3", "SAC", "random_search", "DE"])
    parser.add_argument("--seed", type=int, choices=[0, 1, 2])
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Synthetic-only output must be under Deeplearning/results")
    if args.stage in ("prepare", "all"):
        prepare(output, json.loads(args.config.read_text()))
    if args.stage == "run":
        if args.condition is None or args.method is None or args.seed is None:
            parser.error("run requires --condition, --method and --seed")
        run_one(output, args.condition, args.method, args.seed)
    elif args.stage == "compare":
        if args.condition is None:
            parser.error("compare requires --condition")
        compare(output, args.condition, args.workers)
    elif args.stage == "all":
        for condition in INDICES:
            compare(output, condition, args.workers)
    if args.stage in ("report", "all"):
        from .issue13_report import report
        from .issue13_audit import audit
        load_ready(output)
        verify_protected(output)
        report(output)
        audit(output)


if __name__ == "__main__":
    main()
