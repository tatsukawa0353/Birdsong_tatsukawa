"""Local-only real-recording experiments. Do not publish private_runs artifacts."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from .acoustics import load_local_wav
from .evaluation import descriptors
from .experiments import train_run, write_json
from .model import BirdModel, ROOT
from .model_env import ModelConfig, ModelEnv, AudioDecimator
from .plotting import learning_curves

CONTROL_INDICES = {"A": (0, 2, 3), "B": (0, 1, 2, 3), "C": (0, 1, 2, 3, 4)}
FIXED_INDICES = {"A": (0, 2), "B": (0, 1, 2), "C": (0, 1, 2)}


def run(config, target, output, *, local_permission_confirmed):
    if not local_permission_confirmed:
        raise ValueError("Local research permission must be confirmed before reading the recording")
    private_root = (ROOT/"private_runs").resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(private_root):
        raise ValueError("Recording-derived results must remain inside Deeplearning/private_runs")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    audio, preprocessing = load_local_wav(target, target_rate=50000, return_metadata=True)
    if np.max(np.abs(audio)) < 1e-8:
        raise ValueError("Selected real target is silent; choose an explicit non-silent segment")
    # One calibration using the default A reference, never one gain per condition.
    with BirdModel("A") as model:
        reference = AudioDecimator().process(model.advance(.35))
    target_rms = float(np.sqrt(np.mean(audio**2)))
    gain = float(np.sqrt(np.mean(reference**2))/target_rms)
    write_json(output/"config.json", dict(experiment=config, local_permission_confirmed=True,
                                         output_gain=gain,
                                         calibration="single default-A RMS / target RMS, shared across A/B/C",
                                         target_duration=.35, target_rate=50000,
                                         preprocessing=preprocessing,
                                         target_descriptors=descriptors(audio, 50000)))
    summaries, curves = [], {}
    variant = config.get("force_variant", "linear")
    if variant not in ("fixed", "linear", "power8"):
        raise ValueError("force_variant must be fixed, linear or power8")
    for condition in config["conditions"]:
        indices = FIXED_INDICES[condition] if variant == "fixed" else CONTROL_INDICES[condition]
        settings = ModelConfig(condition=condition, control_indices=indices, output_gain=gain,
                               fe_mapping="power" if variant == "power8" else "linear", diagnostics=True)
        write_json(output/f"condition_{condition}.json", asdict(settings))
        factory = lambda settings=settings: ModelEnv(settings, external_target=audio)
        curves[condition] = []
        for seed in config["seeds"]:
            print(f"Phase 2: condition={condition}, seed={seed}", flush=True)
            summary, entries = train_run(config["algorithm"], seed, factory, factory,
                                         config, output/f"{condition}_{seed}")
            summaries.append(dict(condition=condition, free_controls=len(settings.control_indices), **summary))
            write_json(output/"summary.json", summaries)
            curves[condition].append(([e["steps"] for e in entries], [e["error"] for e in entries]))
    learning_curves(curves, output/"learning_curves.png")
    write_json(output/"interpretation.json", {
        "scope": "model reachability and training efficiency only; no neural-learning inference",
        "limitations": [f"control dimensions depend on the selected force variant: {variant}",
                        "B uses a fixed right external force; verify vibration suppression from saved states",
                        "waveform phase, vocal-tract filter and recording gain are not separately identified",
                        "single common gain calibration does not establish physical pressure units",
                        "small-budget learning failure is not evidence of representational impossibility"]})
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Permission-restricted local recording comparison")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/phase2.json")
    parser.add_argument("--output", type=Path, default=ROOT/"private_runs/phase2")
    parser.add_argument("--local-permission-confirmed", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    previous_umask = os.umask(0o077)
    try:
        run(config, args.target, args.output, local_permission_confirmed=args.local_permission_confirmed)
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    main()
