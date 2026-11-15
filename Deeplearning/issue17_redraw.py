"""Redraw Issue #17 curves separately for 1D, 2D and 3D; no training.

Reads the frozen results/phase1_constant_020s/comparison_curves.csv.
Generates twelve NEW images without changing the existing combined figures,
model artifacts, CSVs, report, manifest, audit, or old results.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .constant020_env import INDICES

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results" / "phase1_constant_020s"
METHODS = ("TD3", "SAC", "DE", "random_search")
SEEDS = (0, 1, 2)
COLORS = {"TD3": "#0072B2", "SAC": "#D55E00",
          "DE": "#009E73", "random_search": "#CC79A7"}
LABELS = {"TD3": "TD3", "SAC": "SAC", "DE": "Differential evolution",
          "random_search": "Random search"}


def redraw(results=DEFAULT_RESULTS):
    """Create separate images without regenerating any learning result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401  (registers the style)

    results = Path(results)
    csv_file = results / "comparison_curves.csv"
    if not csv_file.is_file():
        raise FileNotFoundError(f"Saved Issue #17 curves are required: {csv_file}")
    figures = results / "figures"
    if not figures.is_dir():
        raise FileNotFoundError(f"Expected existing figures folder: {figures}")
    with csv_file.open("r", newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    if not records:
        raise ValueError("No rows found in saved curves CSV")

    grouped = {}
    for row in records:
        key = (row["condition"], row["method"], int(row["seed"]))
        grouped.setdefault(key, []).append(row)

    # Fail safely on incomplete or duplicate records, before writing any image.
    for condition in INDICES:
        for method in METHODS:
            for seed in SEEDS:
                key = condition, method, seed
                if key not in grouped:
                    raise ValueError(f"Missing run {key}")
                points = grouped[key]
                x = [int(p["episodes"]) for p in points]
                expected = list(range(0, 257, 8)) if method in ("TD3", "SAC") else list(range(1, 257))
                if x != expected:
                    raise ValueError(f"Unexpected checkpoint positions or ordering for {key}")
                for p in points:
                    if not (np.isfinite(float(p["E_total"])) and
                            np.isfinite(float(p["best_observed_error"]))):
                        raise ValueError(f"Non-finite error for {key}")
    saved = []
    for condition in INDICES:
        suffix = condition.replace("constant_", "")
        for kind, field in (("learning_curves", "E_total"),
                            ("best_observed", "best_observed_error")):
            for average in (False, True):
                with plt.style.context(["science", "ieee", "no-latex"]), plt.rc_context(
                        {"path.simplify": False}):
                    fig, ax = plt.subplots(figsize=(12.4, 6.5))
                    for method in METHODS:
                        trajectories = [grouped[(condition, method, seed)] for seed in SEEDS]
                        x = np.array([int(p["episodes"]) for p in trajectories[0]])
                        y = np.array([[float(p[field]) for p in run] for run in trajectories])
                        center = y.mean(axis=0) if average else y[0]
                        ax.plot(x, center, color=COLORS[method], label=LABELS[method],
                                linewidth=2.2, alpha=0.95)
                        if average:
                            std = y.std(axis=0, ddof=1)
                            ax.fill_between(x, np.maximum(0, center - std), center + std,
                                            color=COLORS[method], alpha=0.12, linewidth=0)
                    ax.set_xlim(0, 256)
                    ax.set_ylim(bottom=0)  # all actual acoustic errors are nonnegative
                    ax.set_xticks([0, 64, 128, 192, 256])
                    ax.set_xlabel("Training candidates / one-action episodes", fontsize=17)
                    ax.set_ylabel("Best observed acoustic error" if kind == "best_observed"
                                  else "Acoustic error", fontsize=17)
                    ax.tick_params(axis="both", labelsize=14)
                    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03),
                              ncol=4, frameon=False, fontsize=13)
                    # The same set of methods; the vertical scale is now fitted PER
                    # dimensionality rather than shared by three cramped subplots.
                    fig.tight_layout()
                    name = f"{kind}_{suffix}_{'mean_sd' if average else 'seed0'}.png"
                    destination = figures / name
                    if destination.exists():
                        plt.close(fig)
                        raise FileExistsError(
                            f"Figure already exists; refusing to overwrite it: {destination}")
                    fig.savefig(destination, dpi=160, bbox_inches="tight", pad_inches=0.15)
                    plt.close(fig)
                    saved.append(name)
    metadata = figures / "separate_curves_metadata.json"
    if metadata.exists():
        raise FileExistsError(f"Metadata already exists; preserving it: {metadata}")
    data = dict(source_csv=str(csv_file.relative_to(results)),
                source_sha256=hashlib.sha256(csv_file.read_bytes()).hexdigest(),
                images=saved, conditions=list(INDICES), methods=list(METHODS),
                seeds=list(SEEDS), separate_y_axis_per_dimension=True,
                original_combined_figures_unchanged=True,
                no_training_or_evaluation=True,
                actual_curve="RL current deterministic actor; direct methods actual candidate",
                best_curve="best observed as stored; not interchangeable with final candidate",
                plotted_3seed_sd="sample standard deviation with display-only lower band clipped at zero")
    metadata.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"Generated {len(saved)} new separate figures in {figures}")
    for name in saved:
        print(f"  {name}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    redraw(args.results)
