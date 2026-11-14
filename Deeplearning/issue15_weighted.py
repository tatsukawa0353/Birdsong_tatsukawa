"""Issue #15 follow-up: preregistered lambda=0.5 ablation, no overwrite of A/B/C."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import subprocess
import tempfile
import time
from zoneinfo import ZoneInfo

import numpy as np

from .issue14 import train_one, diagnostics
from .issue15_analysis import write_json
from .issue15_report import _spectrogram, controls_with_clear_legend

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "results" / "phase0_fixed_target_20261108"
ABLATION = PARENT / "reward_ablation"
DESTINATION = ABLATION / "weighted_lambda_0p5"
WEIGHT = 0.5
ALGORITHMS = ("PPO", "TD3", "SAC")
SEEDS = (0, 1, 2)
SOURCES = ("sine_env.py", "experiments.py", "acoustics.py", "issue14.py",
           "issue15_weighted.py", "issue15_report.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def individual_records(folder):
    results = []
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            directory = folder / f"{algorithm}_{seed}"
            summary = json.loads((directory / "summary.json").read_text())
            curve = json.loads((directory / "curve.json").read_text())
            assert summary["training_steps"] == 102400
            assert summary["evaluation_reward_terms"] == "combined"
            assert summary["training_reward_terms"] == "weighted"
            assert len(curve) == 51
            assert [point["steps"] for point in curve] == list(range(0, 102401, 2048))
            assert curve[-1]["evaluation_phase"] == "post_final_update"
            assert abs(curve[-1]["error"] - summary["error"]) < 1e-12
            assert abs(summary["metrics"]["log_spectral"] +
                       summary["metrics"]["envelope"] - summary["error"]) < 1e-12
            assert digest(directory / "agent.zip") == summary["model_sha256"]
            assert digest(directory / "rollout.npz") == summary["rollout_sha256"]
            with np.load(directory / "training_trace.npz", allow_pickle=False) as trace:
                actual = trace["rewards"]
                expected = -(trace["log_spectral"] + WEIGHT * trace["envelope"])
                np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
                assert len(actual) == 102400
            results.append({
                "condition": "weighted_lambda_0p5",
                "algorithm": algorithm,
                "seed": seed,
                "E_spec": summary["metrics"]["log_spectral"],
                "E_env": summary["metrics"]["envelope"],
                "E_total": summary["error"],
                "frequency_mae_hz": summary["tracking"]["frequency_mae_hz"],
                "amplitude_mae": summary["tracking"]["amplitude_mae"],
                "training_wall_s": summary["training_wall_s"],
                "evaluation_wall_s": summary["evaluation_wall_s"]
            })
    return results


def draw_learning_curves(all_curves, folder):
    colors = {"combined": "#0072B2", "weighted_lambda_0p5": "#D55E00"}
    labels = {"combined": "A: combined (lambda=1)",
              "weighted_lambda_0p5": "D: weighted (lambda=0.5)"}
    for mean in (False, True):
        with plt.style.context(["science", "ieee", "no-latex"]), plt.rc_context({"path.simplify": False}):
            fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            for ax, algorithm in zip(axes, ALGORITHMS):
                for condition in colors:
                    by_seed = []
                    for seed in SEEDS:
                        matched = [row for row in all_curves
                                   if row["condition"] == condition and row["algorithm"] == algorithm
                                   and int(row["seed"]) == seed]
                        assert len(matched) == 51
                        matched.sort(key=lambda row: int(row["steps"]))
                        by_seed.append(np.array([float(row["E_total"]) for row in matched]))
                    array = np.array(by_seed)
                    x = np.arange(0, 102401, 2048)
                    ax.plot(x, array.mean(axis=0) if mean else array[0],
                            label=labels[condition], color=colors[condition],
                            linestyle="-" if condition=="combined" else "--", linewidth=2.6)
                    if mean:
                        sd = array.std(axis=0, ddof=1)
                        ax.fill_between(x, array.mean(axis=0)-sd, array.mean(axis=0)+sd,
                                        color=colors[condition], alpha=0.12)
                ax.set_ylabel(f"{algorithm} acoustic error", fontsize=19)
                ax.tick_params(labelsize=14)
                ax.legend(fontsize=11, loc="upper right")
            axes[-1].set_xlim(0, 102400)
            axes[-1].set_xlabel("Training environment steps", fontsize=19)
            fig.tight_layout()
            fig.savefig(figures / ("learning_curves_mean_sd.png" if mean else "learning_curves_seed0.png"),
                        dpi=140, bbox_inches="tight")
            plt.close(fig)


def plot_results(folder, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots

    existing = records(ABLATION / "comparison_per_seed.csv")
    rows = existing + [{key: str(value) for key, value in row.items()} for row in summaries]
    fields = ("condition", "algorithm", "seed", "E_spec", "E_env", "E_total",
              "frequency_mae_hz", "amplitude_mae")
    write_csv(folder / "comparison_per_seed.csv",
              [{key: row[key] for key in fields} for row in rows], fields)
    aggregate = []
    for condition in ("combined", "spectral_only", "envelope_only", "weighted_lambda_0p5"):
        for algorithm in ALGORITHMS:
            selected = [row for row in rows
                        if row["condition"] == condition and row["algorithm"] == algorithm]
            assert len(selected) == 3
            record = {"condition": condition, "algorithm": algorithm}
            for key in ("E_spec", "E_env", "E_total", "frequency_mae_hz", "amplitude_mae"):
                vals = np.array([float(row[key]) for row in selected])
                record[key + "_mean"] = float(vals.mean())
                record[key + "_sample_sd"] = float(vals.std(ddof=1))
            aggregate.append(record)
    write_csv(folder / "comparison_aggregate.csv", aggregate, list(aggregate[0]))
    old_curves = records(ABLATION / "comparison_curves.csv")
    new_curves = []
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            curve = json.loads((folder / f"{algorithm}_{seed}" / "curve.json").read_text())
            for point in curve:
                new_curves.append({
                    "condition": "weighted_lambda_0p5", "algorithm": algorithm,
                    "seed": seed, "steps": point["steps"],
                    "E_spec": point["metrics"]["log_spectral"],
                    "E_env": point["metrics"]["envelope"],
                    "E_total": point["error"],
                    "frequency_mae_hz": point["tracking"]["frequency_mae_hz"],
                    "amplitude_mae": point["tracking"]["amplitude_mae"]
                })
    all_curves = old_curves + [{key: str(value) for key, value in c.items()} for c in new_curves]
    write_csv(folder / "comparison_curves.csv", all_curves, list(all_curves[0]))
    figures = folder / "figures"
    figures.mkdir()
    draw_learning_curves(all_curves, folder)
    for algorithm in ALGORITHMS:
        directory = folder / f"{algorithm}_0"
        summary = json.loads((directory / "summary.json").read_text())
        with np.load(directory / "rollout.npz", allow_pickle=False) as archive:
            snapshot = {name: archive[name].copy() for name in archive.files}
        run = dict(summary, _snapshot=snapshot)
        _spectrogram(run, figures / f"{algorithm}_seed0_spectrogram.png")
        controls_with_clear_legend(run, figures / f"{algorithm}_seed0_controls.png")
    return aggregate


def report(folder, aggregates, *, run_wall, report_wall):
    lines = [
        "# Issue #15: lambda=0.5 weighted-reward follow-up", "",
        "Training reward per 10 ms step: r = -(E_spec + 0.5 E_env).",
        "All four conditions share the same final 0.35 s combined evaluation E_spec+E_env.",
        "The independent baseline A/B/C models are reused, not retrained or modified.",
        "This evaluates in-target tracking, not bird-song generalization.", "",
        "## Final policy, 3 seeds (mean +/- sample SD)", "",
        "| Condition | Algorithm | Combined acoustic error | Frequency MAE [Hz] | Amplitude MAE |",
        "| --- | --- | ---: | ---: | ---: |"
    ]
    for rec in aggregates:
        lines.append(
            f"| {rec['condition']} | {rec['algorithm']} | "
            f"{rec['E_total_mean']:.6f} +/- {rec['E_total_sample_sd']:.6f} | "
            f"{rec['frequency_mae_hz_mean']:.3f} +/- {rec['frequency_mae_hz_sample_sd']:.3f} | "
            f"{rec['amplitude_mae_mean']:.6f} +/- {rec['amplitude_mae_sample_sd']:.6f} |")
    lines.extend(["", "## Saved figures", "",
                  "- [Seed-0 learning curves](figures/learning_curves_seed0.png)",
                  "- [3-seed means and SD](figures/learning_curves_mean_sd.png)"])
    for algorithm in ALGORITHMS:
        lines.append(f"- {algorithm}: [spectrogram](figures/{algorithm}_seed0_spectrogram.png), "
                     f"[frequency and amplitude tracking](figures/{algorithm}_seed0_controls.png)")
    lines.extend(["", "## Cost and interpretation", "",
                  f"Nine new weighted runs: {run_wall:.3f} s elapsed wall time with concurrent workers.",
                  f"Report/plot creation: {report_wall:.3f} s.",
                  "Check E_spec and E_env separately, along with f/A MAE. The weighted coefficient "
                  "is not the same as a direct frequency penalty.",
                  "Do not interpret a single lambda as the optimum; all runs share the same fixed target.",
                  "The original Issue-15 report and #14 results remain unchanged."])
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n")


def run(workers=6):
    if not 1 <= workers <= 6:
        raise ValueError("workers must be between 1 and 6")
    if DESTINATION.exists():
        raise FileExistsError(f"Results already exist; preserving them: {DESTINATION}")
    if not (ABLATION / "comparison_curves.csv").exists():
        raise FileNotFoundError("Complete Issue-15 baseline results are required")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "main_deeplearning":
        raise RuntimeError("Use the verified main_deeplearning branch")
    base = json.loads((PARENT / "config.json").read_text())
    for key, expected in (("steps", 102400), ("eval_interval", 2048),
                          ("target", "glide"), ("randomize_training_target", False)):
        if base.get(key) != expected:
            raise ValueError(f"Unexpected baseline setting {key}: {base.get(key)}")
    settings = dict(base, reward_terms="weighted", reward_envelope_weight=WEIGHT)
    original_hash = digest(ABLATION / "comparison_per_seed.csv")
    stage = Path(tempfile.mkdtemp(prefix=".weighted_lambda_0p5_stage_", dir=ABLATION))
    write_json(stage / "config.json", settings)
    manifest = {
        "workflow": "issue15_weighted_lambda_0p5",
        "preregistered_lambda": WEIGHT,
        "created_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "branch": branch,
        "base_config": base,
        "parent_baseline_comparison_sha256": original_hash,
        "training_reward": "negative local E_spec plus 0.5 * local E_env",
        "shared_evaluation": "0.35 s combined E_spec + E_env and frequency/amplitude MAE",
        "source_sha256": {name: digest(ROOT / name) for name in SOURCES},
        "versions": {key: importlib.metadata.version(key)
                     for key in ("numpy", "scipy", "torch", "stable-baselines3", "gymnasium")},
        "baseline_retrained": False,
        "old_results_overwritten": False
    }
    write_json(stage / "manifest.json", manifest)
    diagnostics(stage, settings)
    started = time.perf_counter()
    summaries = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(train_one, algorithm, seed, settings, stage / f"{algorithm}_{seed}")
                   for algorithm in ALGORITHMS for seed in SEEDS]
        for future in as_completed(futures):
            summaries.append(future.result())
            write_json(stage / "summary.json", summaries)
    run_wall = time.perf_counter() - started
    if len(summaries) != 9:
        raise RuntimeError("Nine runs required")
    checks = individual_records(stage)
    assert digest(ABLATION / "comparison_per_seed.csv") == original_hash
    plot_started = time.perf_counter()
    aggregate = plot_results(stage, checks)
    report_wall = time.perf_counter() - plot_started
    report(stage, aggregate, run_wall=run_wall, report_wall=report_wall)
    write_json(stage / "execution.json", {
        "completed_runs": 9,
        "training_steps": sum(run["training_steps"] for run in summaries),
        "evaluation_steps": sum(run["evaluation_steps"] for run in summaries),
        "concurrent_run_wall_s": run_wall,
        "plot_and_report_wall_s": report_wall
    })
    if DESTINATION.exists():
        raise FileExistsError("Destination appeared while training; staged artifacts retained")
    stage.rename(DESTINATION)
    print(f"Weighted lambda=0.5 finished: {DESTINATION}", flush=True)



def redraw_only():
    """Update existing A-vs-D curve images from CSV; no training or new output files."""
    if not DESTINATION.is_dir():
        raise FileNotFoundError(f"Weighted results not found: {DESTINATION}")
    csv_file = DESTINATION / "comparison_curves.csv"
    if not csv_file.is_file():
        raise FileNotFoundError(csv_file)
    figure_dir = DESTINATION / "figures"
    if not figure_dir.is_dir():
        raise FileNotFoundError(figure_dir)
    draw_learning_curves(records(csv_file), figure_dir)
    print("Redrew A (lambda=1) and D (lambda=0.5) only; existing filenames reused.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--redraw-only", action="store_true",
                        help="Redraw only A vs D learning curves from saved data (no training)")
    args = parser.parse_args()
    if args.redraw_only:
        redraw_only()
    else:
        run(args.workers)
