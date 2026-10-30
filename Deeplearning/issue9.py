"""Issue #9 diagnosis, control repair and equal-budget local SAC comparisons.

Synthetic state maps may be shared. Every recording-derived artifact is kept
under private_runs/issue9 or repaired in place under private_runs/phase2.
"""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import wavfile

from .acoustics import load_local_wav
from .experiments import train_run, write_json, rollout
from .model import ROOT, DEFAULT_CONTROLS, build, effective_controls
from .model_env import ModelConfig, ModelEnv
from .phase2 import CONTROL_INDICES, FIXED_INDICES
from .plotting import control_trajectories, comparison, learning_curves, plt
from .voice_diagnostics import replay, summarize, band_audio, rms

PUBLIC = ROOT/"results/issue9"
PRIVATE = ROOT/"private_runs/issue9"


def model_configuration(variant, case, *, gain=1.0):
    return ModelConfig(condition=case,
                       control_indices=FIXED_INDICES[case] if variant == "fixed" else CONTROL_INDICES[case],
                       fe_mapping="power" if variant == "power8" else "linear", fe_power=8,
                       output_gain=gain, diagnostics=True)


def atomic_npz(path, arrays):
    """Preserve old arrays while correcting metadata; never expose private data."""
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temp:
        name = Path(temp.name)
        np.savez_compressed(temp, **arrays)
    os.replace(name, path)


def run_map(config):
    PUBLIC.mkdir(parents=True, exist_ok=True)
    baseline = replay(np.tile(DEFAULT_CONTROLS, (35, 1)), "A")
    reference = rms(band_audio(baseline["audio"])[5000:])
    combinations = list(itertools.product(config["conditions"], config["map_epsilon"],
                                         config["map_ps"], config["map_fe"]))
    rows = []
    for i, (case, epsilon, ps, fe) in enumerate(combinations):
        controls = np.array([epsilon, epsilon, ps, fe, fe])
        recording = replay(np.tile(controls, (35, 1)), case)
        diagnosis = summarize(recording["audio"], reference_band_rms=reference, condition=case,
                              motion_mean=recording["motion_mean"], motion_std=recording["motion_std"])
        rows.append(dict(condition=case, epsilon=epsilon, ps=ps, requested_fe=fe,
                         applied_controls=recording["physical_controls"][0].tolist(), **diagnosis))
        if (i+1) % 24 == 0 or i+1 == len(combinations):
            print(f"Synthetic phonation map: {i+1}/{len(combinations)}", flush=True)
            write_json(PUBLIC/"phonation_map.json", rows)
    write_json(PUBLIC/"map_config.json", dict(settings=config, reference_band_rms=reference,
                                              transient_excluded_s=.1, integration_dt=1e-7,
                                              band_hz=[250, 10000], relative_band_threshold_db=-60,
                                              velocity_std_threshold=1.0,
                                              scope="constant controls for 0.35 s; not a global bifurcation proof"))
    plot_map(config, rows)


def plot_map(config, rows):
    categories = {"below_reference_threshold": 0, "transient_or_intermittent": 1,
                  "sustained_noise_like": 2, "sustained_periodic_like": 3}
    for case in config["conditions"]:
        with plt.style.context(["science", "ieee", "no-latex"]):
            fig, axes = plt.subplots(1, len(config["map_epsilon"]), figsize=(16, 5), sharey=True)
            for ax, epsilon in zip(np.atleast_1d(axes), config["map_epsilon"]):
                matrix = np.zeros((len(config["map_ps"]), len(config["map_fe"])))
                for row in rows:
                    if row["condition"] == case and row["epsilon"] == epsilon:
                        matrix[config["map_ps"].index(row["ps"]), config["map_fe"].index(row["requested_fe"])] = categories[row["acoustic_state"]]
                image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=3,
                                  interpolation="nearest")
                ax.set_xticks(range(len(config["map_fe"])), [f"{v:.3g}" for v in config["map_fe"]], rotation=60)
                ax.set_yticks(range(len(config["map_ps"])), [f"{v:.3g}" for v in config["map_ps"]])
                ax.set_title(f"Case {case}, epsilon={epsilon:.3g}")
                ax.set_xlabel(r"$f_e$ (B: left only)")
            np.atleast_1d(axes)[0].set_ylabel(r"$p_s$")
            fig.colorbar(image, ax=list(np.atleast_1d(axes)), ticks=[0, 1, 2, 3],
                         label="0: below threshold, 1: intermittent, 2: noise-like, 3: periodic-like")
            fig.savefig(PUBLIC/f"phonation_map_{case}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
    # Deliberately retain the entire original external-force range, including off states.
    a = np.linspace(-1, 1, 501)
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, ax = plt.subplots(figsize=(8, 5))
        for power, name in [(1, "Linear"), (8, "Power 8")]:
            ax.plot(a, 4.9e4+(1e7-4.9e4)*((a+1)/2)**power, label=name)
        ax.set(xlabel="Normalized action", ylabel=r"$f_e$", yscale="log")
        ax.legend()
        fig.savefig(PUBLIC/"force_mapping.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def local_target(path, permission_confirmed):
    if not permission_confirmed:
        raise ValueError("Confirm local research permission before reading real audio")
    audio, metadata = load_local_wav(path, target_rate=50000, return_metadata=True)
    previous = json.loads((ROOT/"private_runs/phase2/config.json").read_text())
    gain = previous["output_gain"]
    baseline = replay(np.tile(DEFAULT_CONTROLS, (35, 1)), "A", output_gain=gain)
    reference = rms(band_audio(baseline["audio"])[5000:])
    PRIVATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json(PRIVATE/"target_config.json", dict(preprocessing=metadata, local_permission_confirmed=True,
                                                  output_gain=gain, reference_band_rms=reference,
                                                  calibration="reuse original Phase 2 common gain; no per-case recalibration"))
    return audio, gain, reference


def neutral_checks(config):
    reference = json.loads((PUBLIC/"map_config.json").read_text())["reference_band_rms"]
    rows = []
    for variant, case in itertools.product(config["variants"], config["conditions"]):
        cfg = model_configuration(variant, case)
        env = ModelEnv(cfg, external_target=np.zeros(17500))
        try:
            controls = env.decode(np.zeros(len(env.indices)))
        finally:
            env.close()
        recording = replay(np.tile(controls, (35, 1)), case)
        stats = summarize(recording["audio"], reference_band_rms=reference, condition=case,
                          motion_mean=recording["motion_mean"], motion_std=recording["motion_std"])
        rows.append(dict(variant=variant, condition=case, neutral_controls=controls.tolist(), **stats))
    write_json(PUBLIC/"neutral_controls.json", rows)


def repair_history(reference, gain):
    rows = []
    for case in "ABC":
        for seed in (0, 1, 2):
            path = ROOT/f"private_runs/phase2/{case}_{seed}/rollout.npz"
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key].copy() for key in data.files}
            old = arrays["physical_controls"].copy()
            requested = arrays.get("requested_controls", old).copy()
            applied = np.array([effective_controls(c, case) for c in old])
            reproduced = replay(applied, case, output_gain=gain)
            np.testing.assert_allclose(reproduced["audio"], arrays["audio"], rtol=1e-10, atol=1e-12)
            arrays.update(physical_controls=applied, requested_controls=requested,
                          motion_mean=reproduced["motion_mean"], motion_std=reproduced["motion_std"])
            audio_hash = hashlib.sha256(arrays["audio"].tobytes()).hexdigest()
            atomic_npz(path, arrays)
            control_trajectories(applied, path.parent/"controls.png")
            error = json.loads((path.parent/"summary.json").read_text())["error"]
            stats = summarize(arrays["audio"], reference_band_rms=reference, condition=case,
                              motion_mean=reproduced["motion_mean"], motion_std=reproduced["motion_std"],
                              error=error, inspect_display=True)
            rows.append(dict(condition=case, seed=seed, controls_corrected=bool(np.any(old != applied)),
                             audio_sha256=audio_hash, original_requested_controls_retained=True, **stats))
    write_json(PRIVATE/"historical_diagnostics.json", rows)
    print("Diagnosed and repaired 9 historical local rollouts", flush=True)


def known_pipeline(target, gain, reference):
    rows = []
    for case in "ABC":
        cfg = ModelConfig(condition=case, control_indices=FIXED_INDICES[case], output_gain=gain, diagnostics=True)
        env = ModelEnv(cfg, external_target=target)
        try:
            result, audio, _ = rollout(env, lambda obs, e: e.encode(DEFAULT_CONTROLS))
            controls = np.array(env.applied_control_history)
            independent = replay(controls, case, output_gain=gain)
            np.testing.assert_array_equal(audio, independent["audio"])
            out = PRIVATE/f"known_{case}"
            out.mkdir(exist_ok=True)
            np.savez_compressed(out/"rollout.npz", target=target, audio=audio, sample_rate=50000,
                                physical_controls=controls)
            comparison(target, audio, 50000, out/"spectrogram.png")
            control_trajectories(controls, out/"controls.png")
            wavfile.write(out/"generated.wav", 50000, (audio/max(1, np.max(abs(audio)))).astype(np.float32))
            stats = summarize(audio, reference_band_rms=reference, condition=case,
                              motion_mean=independent["motion_mean"], motion_std=independent["motion_std"],
                              error=result["error"], inspect_display=True)
            rows.append(dict(condition=case, pipeline_replay_identical=True, **stats))
        finally:
            env.close()
    write_json(PRIVATE/"known_pipeline.json", rows)


def run_training(config, target, gain, reference):
    rows = []
    for variant in config["variants"]:
        curves = {}
        for case in config["conditions"]:
            cfg = model_configuration(variant, case, gain=gain)
            factory = lambda cfg=cfg: ModelEnv(cfg, external_target=target)
            curves[case] = []
            for seed in config["seeds"]:
                out = PRIVATE/f"experiments/{variant}/{case}_{seed}"
                out.mkdir(parents=True, exist_ok=True)
                expected = dict(model=asdict(cfg), experiment=config, seed=seed)
                config_file = out/"config.json"
                # Serialize once so tuple/list representation is stable across resume.
                expected = json.loads(json.dumps(expected))
                if config_file.exists() and json.loads(config_file.read_text()) != expected:
                    raise ValueError("Existing run has different settings; choose a fresh output directory")
                write_json(config_file, expected)
                if (out/"summary.json").exists() and (out/"curve.json").exists() and (out/"rollout.npz").exists():
                    result = json.loads((out/"summary.json").read_text())
                    entries = json.loads((out/"curve.json").read_text())
                else:
                    print(f"Issue 9 SAC: {variant}, Case {case}, seed {seed}", flush=True)
                    result, entries = train_run("SAC", seed, factory, factory, config, out)
                with np.load(out/"rollout.npz", allow_pickle=False) as data:
                    controls, audio = data["physical_controls"], data["audio"]
                    diagnosis = summarize(audio, reference_band_rms=reference, condition=case,
                                          motion_mean=data["motion_mean"], motion_std=data["motion_std"],
                                          error=result["error"], inspect_display=True, numerical_failure=result["failed"])
                    if case == "B":
                        np.testing.assert_array_equal(controls[:, 4], np.full(len(controls), 1e7))
                    control_trajectories(controls, out/"controls.png")
                    wavfile.write(out/"generated.wav", 50000, (audio/max(1, np.max(abs(audio)))).astype(np.float32))
                write_json(out/"diagnostics.json", diagnosis)
                rows.append(dict(variant=variant, condition=case, free_controls=len(cfg.control_indices), **result,
                                 diagnosis=diagnosis))
                write_json(PRIVATE/"comparison.json", rows)
                curves[case].append(([e["steps"] for e in entries], [e["error"] for e in entries]))
        learning_curves(curves, PRIVATE/f"learning_curves_{variant}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["map", "neutral", "history", "train", "report", "all"])
    parser.add_argument("--config", type=Path, default=ROOT/"configs/issue9.json")
    parser.add_argument("--target", type=Path, default=ROOT/"sample_song/14412_0.35s.wav")
    parser.add_argument("--local-permission-confirmed", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage in ("map", "neutral", "history", "train", "all"):
        build()
    if args.stage in ("map", "all"):
        run_map(config)
    if args.stage in ("map", "neutral", "all"):
        neutral_checks(config)
    if args.stage in ("history", "train", "all"):
        previous = os.umask(0o077)
        try:
            target, gain, reference = local_target(args.target, args.local_permission_confirmed)
            write_json(PRIVATE/"experiment_config.json", config)
            if args.stage in ("history", "all"):
                repair_history(reference, gain)
                known_pipeline(target, gain, reference)
            if args.stage in ("train", "all"):
                run_training(config, target, gain, reference)
        finally:
            os.umask(previous)
    if args.stage in ("report", "all"):
        from .issue9_report import main as report
        report()


if __name__ == "__main__":
    main()
