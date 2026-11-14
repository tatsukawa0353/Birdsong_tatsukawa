"""Read existing Issue 14 JSON only: no learning, policy loading or evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots


ALGORITHMS = ("PPO", "TD3", "SAC")
COLORS = {"PPO": "#0072B2", "TD3": "#009E73", "SAC": "#D55E00"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(control, output):
    control, output = Path(control), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    sources, curves, rows, selected = {}, {}, [], []
    for algorithm in ALGORITHMS:
        curves[algorithm] = []
        for seed in range(3):
            directory = control/f"{algorithm}_{seed}"
            records = {}
            for filename in ("curve.json", "best.json", "summary.json"):
                p = directory/filename
                records[filename] = json.loads(p.read_text())
                sources[str(p.relative_to(control))] = hashlib.sha256(p.read_bytes()).hexdigest()
            curve = records["curve.json"]
            assert [p["steps"] for p in curve] == list(range(0, 102401, 2048))
            curves[algorithm].append(curve)
            for point in curve:
                row = dict(algorithm=algorithm, seed=seed, steps=point["steps"],
                           E_spec=point["metrics"]["log_spectral"], E_env=point["metrics"]["envelope"],
                           E_total=point["error"], frequency_mae_hz=point["tracking"]["frequency_mae_hz"],
                           amplitude_mae=point["tracking"]["amplitude_mae"])
                assert np.isclose(row["E_total"], row["E_spec"]+row["E_env"], rtol=0, atol=1e-14)
                rows.append(row)
                if point["steps"] in (0, 20480, 51200, 102400):
                    selected.append(dict(role="scheduled", **row))
            best = records["best.json"]
            selected.append(dict(role="recorded_best", algorithm=algorithm, seed=seed, steps=best["steps"],
                E_spec=best["metrics"]["log_spectral"], E_env=best["metrics"]["envelope"], E_total=best["error"],
                frequency_mae_hz=best["tracking"]["frequency_mae_hz"], amplitude_mae=best["tracking"]["amplitude_mae"]))
    aggregates = []
    for name in ALGORITHMS:
        for step in range(0, 102401, 2048):
            values = [r for r in rows if r["algorithm"]==name and r["steps"]==step]
            row = dict(algorithm=name, steps=step)
            for field in ("E_spec", "E_env", "E_total", "frequency_mae_hz", "amplitude_mae"):
                row[field+"_mean"] = float(np.mean([r[field] for r in values]))
                row[field+"_sample_sd"] = float(np.std([r[field] for r in values], ddof=1))
            aggregates.append(row)
    for mean in (False, True):
        with plt.style.context(["science", "ieee", "no-latex"]), plt.rc_context({"path.simplify": False}):
            fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            for column, (field, label) in enumerate((("log_spectral", "Spectral error"), ("envelope", "Envelope error"), ("total", "Total error"))):
                for name in ALGORITHMS:
                    array = np.array([[p["metrics"][field] for p in curve] for curve in curves[name]])
                    x = [p["steps"] for p in curves[name][0]]
                    values = array.mean(axis=0) if mean else array[0]
                    axes[column].plot(x, values, color=COLORS[name], label=name, linewidth=2)
                    if mean:
                        sd = array.std(axis=0, ddof=1)
                        axes[column].fill_between(x, values-sd, values+sd, color=COLORS[name], alpha=.15)
                axes[column].set_ylabel(label, fontsize=20)
                axes[column].tick_params(labelsize=15)
                axes[column].legend(fontsize=15, loc="upper right")
            axes[-1].set_xlabel("Training environment steps", fontsize=20)
            axes[-1].set_xlim(0, 102400)
            fig.tight_layout()
            fig.savefig(output/("components_mean_sd.png" if mean else "components_seed0.png"), dpi=140, bbox_inches="tight")
            plt.close(fig)
    write_csv(output/"components_per_seed.csv", rows)
    write_csv(output/"components_aggregate.csv", aggregates)
    write_csv(output/"selected_points.csv", selected)
    observations = [r for r in aggregates if (r["algorithm"],r["steps"]) in (("TD3",2048),("TD3",102400),("SAC",30720),("SAC",102400))]
    result = dict(input_sha256=sources, added_training_steps=0,
                  added_policy_evaluation_steps=0, source="saved curve/best/summary JSON only",
                  mean_uses_three_seeds=True, sd_ddof=1, smoothing=False,
                  selected_observations=observations, analysis_wall_s=time.perf_counter()-start)
    # Exact full commit is supplied by the orchestrator; no branch lookup or
    # policy evaluation is necessary for this JSON-only analysis.
    write_json(output/"analysis.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.control, args.output)
