"""Local-only, matched Phase 2 comparison; never publish this runner's output."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import wavfile

from .acoustics import load_local_wav
from .experiments import train_run, write_json
from .model import ROOT, DEFAULT_CONTROLS, build
from .model_env import ModelEnv
from .objectives import BalancedConfig, BalancedObjective
from .phase2 import model_configuration, private_output
from .plotting import control_trajectories, single_spectrogram
from .voice_diagnostics import replay, summarize, band_audio, rms

REWARDS = {"baseline": dict(loss_kind="baseline", reward_timing="context"),
           "balanced_aligned": dict(loss_kind="balanced", reward_timing="aligned")}
CORE_FILES = ("model.py", "native/model.cpp", "model_env.py", "acoustics.py", "objectives.py",
              "experiments.py", "phase2.py", "plotting.py", "voice_diagnostics.py")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text())


def validate_protocol(config):
    if (config["algorithms"] != ["TD3", "SAC"] or config["conditions"] != ["A", "B", "C"]
            or config["profiles"] != list(REWARDS) or config["seeds"] != [0, 1, 2]
            or config["force_variant"] != "power8"):
        raise ValueError("Issue 11 uses the declared TD3/SAC, A/B/C, two-profile, three-seed power8 protocol")
    if config["steps"] != 1050 or config["extension_steps"] != 4200 or config["eval_interval"] != 350:
        raise ValueError("Keep the declared Issue 11 budgets and evaluation interval")
    if config["workers"] not in (1, 2, 3):
        raise ValueError("Use one to three separate worker processes")


def prepare(config, target_path, output, *, permission_confirmed):
    if not permission_confirmed:
        raise ValueError("Confirm local research permission before reading recording data")
    output = private_output(output)
    validate_protocol(config)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    target, metadata = load_local_wav(target_path, target_rate=50000, return_metadata=True)
    if rms(target) <= 1e-8:
        raise ValueError("The selected local target is silent")
    reference = replay(np.tile(DEFAULT_CONTROLS, (35, 1)), "A")["audio"]
    gain = rms(reference)/rms(target)
    band_reference = rms(band_audio(reference/gain)[5000:])
    model_hashes = {p: sha((ROOT/p).read_bytes()) for p in CORE_FILES}
    expected = dict(experiment=config, preprocessing=metadata, target_sha256=sha(target.tobytes()),
                    source_sha256=sha(Path(target_path).read_bytes()), output_gain=gain,
                    reference_band_rms=band_reference, shared_loss_scale=max(float(abs(target).max()), 1e-8),
                    local_permission_confirmed=True, balanced_config=asdict(BalancedConfig()),
                    core_sha256=model_hashes,
                    calibration="single default-A full-episode RMS / target RMS; shared across every run",
                    base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    path = output/"protocol.json"
    if path.exists() and json_read(path) != expected:
        raise ValueError("Existing Issue 11 protocol differs; use a fresh private output directory")
    write_json(path, expected)
    np.savez_compressed(output/"target.npz", audio=target, sample_rate=50000)
    obj = BalancedObjective(target, 50000, scale=expected["shared_loss_scale"])
    np.savez_compressed(output/"target_frames.npz", ends=obj.ends, weights=obj.weights,
                        voiced=obj.voiced, envelope=obj.target_features["envelope"],
                        power=obj.target_features["power"], energy_reference=obj.energy_reference)
    wavfile.write(output/"target.wav", 50000, (target/max(1, float(abs(target).max()))).astype(np.float32))
    single_spectrogram(target, 50000, output/"target.png")
    return output


def activity_diagnostics(target, audio, *, scale):
    """Target-only mask and timing/amplitude diagnostics; no new reward term."""
    obj = BalancedObjective(target, 50000, scale=scale)
    target_env = obj.target_features["envelope"]
    generated_env = obj._features(audio, len(obj.ends))["envelope"]
    threshold = .01*float(target_env.max())
    observed = (generated_env >= threshold) & (float(target_env.max()) > 1e-8)
    active = obj.voiced
    w = obj.weights

    def fraction(values, subset):
        mass = w[subset].sum()
        return float(np.sum(w[subset]*values[subset])/mass) if mass > 0 else None

    under = np.maximum(1-generated_env/np.maximum(target_env, 1e-12), 0)
    frames = dict(end_s=(obj.ends/50000).tolist(), weights=w.tolist(),
                  target_active=active.tolist(), generated_active=observed.tolist(),
                  target_envelope=target_env.tolist(), generated_envelope=generated_env.tolist(),
                  loss_parts=obj.frame_parts(audio).tolist(), part_names=list(obj.names))
    result = dict(target_active_fraction=float(w[active].sum()), generated_active_fraction=float(w[observed].sum()),
                  missing_active_fraction=fraction((~observed).astype(float), active),
                  false_voice_quiet_fraction=fraction(observed.astype(float), ~active),
                  activity_disagreement_fraction=float(w[active != observed].sum()),
                  active_underproduction=fraction(under, active),
                  quiet_output_relative=fraction(generated_env/obj.energy_reference, ~active),
                  quiet_duration_s=float(w[~active].sum()*.35),
                  target_mask_rule="target-only >=1% maximum envelope, maximum >1e-8",
                  generated_mask_rule="same absolute target threshold; never normalized by generated peak")
    return result, frames


def run_directory(output, algorithm, steps, case, profile, seed):
    return Path(output)/f"{algorithm}_{steps}/{case}/{profile}/seed_{seed}"


def train_one(output, algorithm, steps, case, profile, seed):
    os.umask(0o077)
    output = private_output(output)
    protocol = json_read(output/"protocol.json")
    config = protocol["experiment"]
    with np.load(output/"target.npz", allow_pickle=False) as data:
        target = data["audio"].copy()
    settings = dict(config, steps=steps, **REWARDS[profile])
    cfg = model_configuration(settings, case, gain=protocol["output_gain"])
    out = run_directory(output, algorithm, steps, case, profile, seed)
    out.mkdir(parents=True, exist_ok=True)
    expected = json.loads(json.dumps(dict(settings=settings, model=asdict(cfg), algorithm=algorithm, seed=seed,
                                          profile=profile, target_sha256=protocol["target_sha256"],
                                          core_sha256=protocol["core_sha256"])))
    if (out/"config.json").exists() and json_read(out/"config.json") != expected:
        raise ValueError("Existing local run settings differ")
    write_json(out/"config.json", expected)
    if all((out/name).exists() for name in ("summary.json", "curve.json", "rollout.npz", "agent.zip", "step_metrics.json")):
        result = json_read(out/"summary.json")
    else:
        print(f"Issue 11: {algorithm}, {steps} steps, Case {case}, {profile}, seed {seed}", flush=True)
        create = lambda: ModelEnv(cfg, external_target=target)
        result, _ = train_run(algorithm, seed, create, create, settings, out)
    if not (out/"diagnostics.json").exists():
        with np.load(out/"rollout.npz", allow_pickle=False) as data:
            audio, controls = data["audio"], data["physical_controls"]
            if case == "B":
                np.testing.assert_array_equal(controls[:, 4], np.full(len(controls), 1e7))
            diagnosis = summarize(audio, reference_band_rms=protocol["reference_band_rms"], condition=case,
                                  motion_mean=data["motion_mean"], motion_std=data["motion_std"],
                                  error=result["error"], inspect_display=len(audio) >= 2000,
                                  numerical_failure=result["failed"])
            if len(audio) == len(target) and not result["failed"]:
                activity, frames = activity_diagnostics(target, audio, scale=protocol["shared_loss_scale"])
                diagnosis["target_activity"] = activity
                write_json(out/"activity_frames.json", frames)
            control_trajectories(controls, out/"controls.png")
            wavfile.write(out/"generated.wav", 50000,
                          (audio/max(1, float(abs(target).max()))).astype(np.float32))
        write_json(out/"diagnostics.json", diagnosis)
    return dict(algorithm=algorithm, budget=steps, condition=case, profile=profile, seed=seed,
                free_controls=len(cfg.control_indices), relative_path=str(out.relative_to(output)),
                result=result, diagnosis=json_read(out/"diagnostics.json"))


def cohort(output, algorithm, steps, cases):
    protocol = json_read(Path(output)/"protocol.json")
    config = protocol["experiment"]
    jobs = [(output, algorithm, steps, case, profile, seed)
            for case, profile, seed in itertools.product(cases, config["profiles"], config["seeds"])]
    if not jobs:
        return []
    # Separate processes keep Torch/NumPy RNGs independent. Submission and
    # aggregation follow a fixed order even when individual jobs finish earlier.
    if config["workers"] == 1:
        rows = [train_one(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=config["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(train_one, *job) for job in jobs]
            rows = [future.result() for future in futures]
    write_json(Path(output)/f"{algorithm}_{steps}/summary.json", rows)
    print(f"Completed Issue 11 cohort: {algorithm}, {steps} steps, {len(rows)} runs", flush=True)
    return rows


def extension_cases(output, algorithm, initial):
    config = json_read(Path(output)/"protocol.json")["experiment"]
    decisions = []
    for case in config["conditions"]:
        group = [r for r in initial if r["condition"] == case]
        failed_evaluations = sum(bool(e["failed"]) for r in group
                                 for e in json_read(Path(output)/r["relative_path"]/"curve.json"))
        failures = sum(r["result"]["training_numerical_failures"] for r in group)
        eligible = (len(group) == 6 and failures == 0 and failed_evaluations == 0
                    and all(not r["result"]["failed"] for r in group))
        decisions.append(dict(condition=case, eligible=eligible, completed_initial_runs=len(group),
                              training_numerical_failures=failures, failed_checkpoints=failed_evaluations))
    write_json(Path(output)/f"{algorithm}_extension_decision.json", dict(rule=config["extension_rule"], cases=decisions))
    return [r["condition"] for r in decisions if r["eligible"]]


def train(output):
    output = private_output(output)
    protocol = json_read(output/"protocol.json")
    config = protocol["experiment"]
    validate_protocol(config)
    current_hashes = {p: sha((ROOT/p).read_bytes()) for p in CORE_FILES}
    if current_hashes != protocol["core_sha256"]:
        raise ValueError("Core source changed since preparation; use a new private experiment directory")
    with np.load(output/"target.npz", allow_pickle=False) as data:
        if sha(data["audio"].tobytes()) != protocol["target_sha256"]:
            raise ValueError("Cached target does not match the prepared protocol")
    rows = []
    for algorithm in config["algorithms"]:
        initial = cohort(output, algorithm, config["steps"], config["conditions"])
        rows.extend(initial)
        cases = extension_cases(output, algorithm, initial)
        rows.extend(cohort(output, algorithm, config["extension_steps"], cases))
        write_json(output/"summary.json", rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "report", "all"))
    parser.add_argument("--config", type=Path, default=ROOT/"configs/issue11.json")
    parser.add_argument("--target", type=Path, default=ROOT/"sample_song/14412_0.35s.wav")
    parser.add_argument("--output", type=Path, default=ROOT/"private_runs/issue11")
    parser.add_argument("--local-permission-confirmed", action="store_true")
    args = parser.parse_args()
    output = private_output(args.output)
    previous = os.umask(0o077)
    try:
        if args.stage in ("prepare", "all"):
            build()
            prepare(json_read(args.config), args.target, output, permission_confirmed=args.local_permission_confirmed)
        if args.stage in ("train", "all"):
            if not args.local_permission_confirmed:
                raise ValueError("Confirm permission before training with the cached recording")
            train(output)
        if args.stage in ("report", "all"):
            from .issue11_report import main as report
            report(output)
    finally:
        os.umask(previous)


if __name__ == "__main__":
    main()
