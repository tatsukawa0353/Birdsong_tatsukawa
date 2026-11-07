"""Figures and report for the constant-control Issue #13 experiment.

Only the new output directory is written.  Learning-curve coordinates are read
directly from curve.json, without interpolation, smoothing, or resampling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 registers the requested styles

from .plotting import (CMAP, MAX_HZ, NOVERLAP, NPERSEG, TARGET_FS, VMAX,
                       VMIN, WINDOW_TYPE, draw_spectrogram)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = ("constant_1d", "constant_2d")
METHODS = ("TD3", "SAC", "random_search", "DE")
SEEDS = (0, 1, 2)
LABELS = {"TD3": "TD3", "SAC": "SAC", "random_search": "Random search", "DE": "Differential evolution"}
COLORS = {"TD3": "#0072B2", "SAC": "#D55E00", "random_search": "#009E73", "DE": "#CC79A7"}
ACCOUNT_FIELDS = ("decision_steps", "episodes", "integration_s", "native_steps", "wall_s", "failures")


def _read(path):
    return json.loads(Path(path).read_text())


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value, precision=7):
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.{precision}g}" if np.isfinite(value) else str(value)


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1 if len(values) > 1 else 0))


def _mean_std_text(values):
    mean, std = _mean_std(values)
    return f"{mean:.7g} ± {std:.3g}"


def _table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |" for row in rows),
    ])


def _csv(path, fields, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_runs(output):
    runs = []
    output = Path(output)
    for condition in CONDITIONS:
        for method in METHODS:
            for seed in SEEDS:
                directory = output / "runs" / condition / method / f"seed_{seed}"
                path = directory / "summary.json"
                if not path.exists():
                    continue
                summary = _read(path)
                for key, expected in (("condition", condition), ("method", method), ("seed", seed)):
                    if summary.get(key) != expected:
                        raise ValueError(f"{path}: {key} disagrees with its result directory")
                for stage in ("initial", "final", "best_observed"):
                    value = float(summary[stage]["acoustic_error"])
                    if not np.isfinite(value) or value < 0:
                        raise ValueError(f"{path}: invalid {stage} acoustic_error")
                summary = dict(summary, _directory=directory, _curve=_read(directory / "curve.json"))
                runs.append(summary)
    return runs


def _curve_values(run):
    curve = run["_curve"]
    x = np.asarray([row["episodes"] for row in curve], dtype=float)
    y = np.asarray([row["acoustic_error"] for row in curve], dtype=float)
    if not np.array_equal(x, np.arange(257)):
        raise ValueError(f"{run['_directory']}: expected the initial point and trials 1 through 256 exactly once")
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError(f"{run['_directory']}: invalid learning-curve acoustic error")
    return x, y


def _legend_overlaps_data(ax, legend, series):
    """Check segment bounding boxes against the actual legend after layout."""
    rectangle = legend.get_window_extent(ax.figure.canvas.get_renderer())
    for x, y in series:
        points = ax.transData.transform(np.column_stack((x, y)))
        for left, right in zip(points[:-1], points[1:]):
            if (max(left[0], right[0]) >= rectangle.x0 and min(left[0], right[0]) <= rectangle.x1
                    and max(left[1], right[1]) >= rectangle.y0 and min(left[1], right[1]) <= rectangle.y1):
                return True
    return False


def plot_learning_curves(output, *, seed=0):
    """Draw exactly four curves for one seed per condition with common limits.

    The legend stays inside the upper right of each axes.  Upper whitespace is
    intentional: with 30 pt legend text it prevents the legend obscuring data.
    """
    if seed not in SEEDS:
        raise ValueError("Select seed 0, 1 or 2")
    output = Path(output)
    runs = [run for run in _load_runs(output) if run["seed"] == seed]
    expected = {(condition, method) for condition in CONDITIONS for method in METHODS}
    if {(run["condition"], run["method"]) for run in runs} != expected:
        raise ValueError(f"Both conditions need all four completed runs for seed {seed} before plotting")
    series = {(run["condition"], run["method"]): _curve_values(run) for run in runs}
    maximum = max(float(y.max()) for _, y in series.values())
    upper = .6
    if maximum > upper:
        raise ValueError("The requested upper limit would clip a recorded acoustic error")
    # Separate the two zero tick labels at the lower-left corner without
    # changing, shifting, or clipping any measured acoustic-error values.
    ylim = (-upper * .03, upper)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        seed=seed, methods=list(METHODS), coordinates="unmodified curve.json episodes/acoustic_error",
        interpolation=False, smoothing=False, path_simplify=False,
        xlim=[0, 256], ylim=list(ylim), legend_labels=[LABELS[method] for method in METHODS],
        figsize_inches=[16, 12], style=["science", "ieee", "no-latex"],
        axis_label_fontsize=46, tick_fontsize=30, legend_fontsize=30,
        linewidth=2.8, linestyle="-", solid_capstyle="round", solid_joinstyle="round",
        antialiased=True, source_file_sha256=_sha(Path(__file__)),
        legend_position="inside upper right", title=None,
        threshold_line=False, uncertainty_band=False, figures={},
    )
    points = []
    with plt.style.context(["science", "ieee", "no-latex"]), plt.rc_context({"path.simplify": False}):
        for condition in CONDITIONS:
            fig, ax = plt.subplots(figsize=(16, 12))
            try:
                drawn = []
                for method in METHODS:
                    x, y = series[(condition, method)]
                    ax.plot(x, y, color=COLORS[method], linewidth=2.8, label=LABELS[method],
                            linestyle="-", solid_capstyle="round", solid_joinstyle="round",
                            antialiased=True)
                    drawn.append((x, y))
                    run = next(r for r in runs if r["condition"] == condition and r["method"] == method)
                    for row in run["_curve"]:
                        points.append(dict(condition=condition, method=method, seed=seed,
                                           episodes=row["episodes"], acoustic_error=row["acoustic_error"],
                                           candidate_error=row.get("candidate_error")))
                ax.set_xlim(0, 256)
                ax.set_ylim(*ylim)
                ax.set_xlabel("Number of trials", fontsize=46)
                ax.set_ylabel("Acoustic error", fontsize=46)
                ax.set_xticks([0, 64, 128, 192, 256])
                ax.tick_params(labelsize=30, width=2, length=9, direction="in", top=False, right=False)
                ax.yaxis.get_offset_text().set_fontsize(30)
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                legend = ax.legend(loc="upper right", fontsize=30, frameon=True,
                                   facecolor="white", edgecolor="none", framealpha=1)
                fig.tight_layout()
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                x_zero = next(tick.label1 for tick in ax.xaxis.get_major_ticks() if tick.get_loc() == 0)
                y_zero = next(tick.label1 for tick in ax.yaxis.get_major_ticks() if tick.get_loc() == 0)
                if x_zero.get_window_extent(renderer).overlaps(y_zero.get_window_extent(renderer)):
                    raise RuntimeError("The origin tick labels overlap; increase the shared lower margin")
                if _legend_overlaps_data(ax, legend, drawn):
                    raise RuntimeError("The legend overlaps measured data; increase the shared upper margin")
                box = legend.get_window_extent(fig.canvas.get_renderer())
                axes_box = ax.get_window_extent()
                if not (box.x0 >= axes_box.x0 and box.x1 <= axes_box.x1
                        and box.y0 >= axes_box.y0 and box.y1 <= axes_box.y1):
                    raise RuntimeError("The legend must remain inside the axes")
                filename = (f"{condition}_learning_curves.png" if seed == 0 else
                            f"{condition}_seed_{seed}_learning_curves.png")
                fig.savefig(figures / filename, dpi=150, bbox_inches="tight", pad_inches=.1)
                metadata["figures"][condition] = dict(
                    path=f"figures/{filename}", legend_overlaps_data=False, legend_inside_axes=True,
                    origin_zero_ticklabels_overlap=False,
                    path_vertex_counts={method: len(line.get_path().vertices) for method, line in zip(METHODS, ax.lines)},
                    path_should_simplify={method: bool(line.get_path().should_simplify) for method, line in zip(METHODS, ax.lines)},
                    source_sha256={method: _sha(output / "runs" / condition / method / f"seed_{seed}" / "curve.json")
                                   for method in METHODS},
                )
            finally:
                plt.close(fig)
    metadata_name = "plot_metadata.json" if seed == 0 else f"plot_metadata_seed_{seed}.json"
    _write_json(figures / metadata_name, metadata)
    _csv(output / f"seed{seed}_learning_curve_points.csv",
         ["condition", "method", "seed", "episodes", "acoustic_error", "candidate_error"], points)
    return metadata


def plot_all_seed_curves(output):
    """Save separate 1D/2D comparisons for every recorded seed, without training."""
    output = Path(output)
    figures = []
    for seed in SEEDS:
        metadata = plot_learning_curves(output, seed=seed)
        for condition, entry in metadata["figures"].items():
            path = entry["path"]
            if seed == 0:
                # Keep the report's original seed-0 filenames and also provide
                # consistently named copies alongside seeds 1 and 2.
                path = f"figures/{condition}_seed_0_learning_curves.png"
                shutil.copy2(output / entry["path"], output / path)
            figures.append(dict(condition=condition, seed=seed, path=path,
                                png_sha256=_sha(output / path),
                                source_sha256=entry["source_sha256"]))
    _write_json(output / "figures/all_seed_plot_metadata.json",
                dict(seeds=list(SEEDS), ylim=metadata["ylim"], linestyle="-",
                     source_file_sha256=_sha(Path(__file__)), figures=figures,
                     numerical_smoothing=False, interpolation=False))
    lines = ["# seed別の誤差線図", "",
             "各図は1つのseedの4手法。全実線・線幅2.8・縦軸上限0.6・同じ表示範囲。"
             "実測値を変更せず、線端と接合部を丸めて描画した。平均化・数値平滑化・補間は行っていない。", "",
             "| seed | 一定1変数 | 一定2変数 |", "| --- | --- | --- |"]
    for seed in SEEDS:
        lines.append(f"| {seed} | [PNG](constant_1d_seed_{seed}_learning_curves.png) | "
                     f"[PNG](constant_2d_seed_{seed}_learning_curves.png) |")
    lines += ["", "横軸は訓練エピソード／候補数。各曲線257点は対応する "
              "`runs/<condition>/<method>/seed_<n>/curve.json` の記録値。"
              "元のseed 0用ファイル名も保持した。", ""]
    (output / "figures/SEED_CURVES.md").write_text("\n".join(lines))
    return figures


def presentation_spectrograms(output):
    """Save title-free final seed-0 comparisons without altering run artifacts.

    These figures use only saved final audio.  No model integration, policy
    evaluation, or recomputation of the primary acoustic error is performed.
    """
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        seed=0, episodes=256, source_stage="final", axes_titles=["", ""],
        top_panel="Target", bottom_panel="Generated", xlabel="Time [s]",
        axis_label_fontsize=46, tick_fontsize=30,
        style=["science", "ieee", "no-latex"],
        primary_acoustic_error_recomputed=False, added_integration_s=0,
        display_settings=dict(target_fs=TARGET_FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                              window=WINDOW_TYPE, cmap=CMAP, vmin=VMIN, vmax=VMAX,
                              max_hz=MAX_HZ, complete_windows_only=True,
                              independent_peak_normalization="display only"),
        figures=[],
    )
    with plt.style.context(["science", "ieee", "no-latex"]):
        for condition in CONDITIONS:
            for method in METHODS:
                directory = output / "runs" / condition / method / "seed_0"
                source = directory / "rollout.npz"
                summary_path = directory / "summary.json"
                summary = _read(summary_path)
                if summary["final"]["episodes"] != 256:
                    raise ValueError(f"{summary_path}: presentation figures require the final trial-256 record")
                hashes = dict(rollout=_sha(source), summary=_sha(summary_path))
                with np.load(source, allow_pickle=False) as saved:
                    target = np.asarray(saved["target"], dtype=float)
                    audio = np.asarray(saved["audio"], dtype=float)
                    rate = int(saved["rate"].item())
                fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
                filename = f"{condition}_{method}_final_spectrogram.png"
                try:
                    for ax, wave in zip(axes, (target, audio)):
                        draw_spectrogram(ax, wave, rate)
                        ax.set_title("")
                        ax.set_ylabel("Frequency [Hz]", fontsize=46)
                        ax.tick_params(labelsize=30)
                    axes[-1].set_xlabel("Time [s]", fontsize=46)
                    fig.tight_layout()
                    fig.savefig(figures / filename, dpi=150, bbox_inches="tight", pad_inches=.1)
                finally:
                    plt.close(fig)
                if _sha(source) != hashes["rollout"] or _sha(summary_path) != hashes["summary"]:
                    raise RuntimeError("A presentation spectrogram's source changed during rendering")
                metadata["figures"].append(dict(
                    condition=condition, method=method, seed=0, episodes=256,
                    path=f"figures/{filename}", png_sha256=_sha(figures / filename),
                    source_rollout=str(source.relative_to(output)), source_rollout_sha256=hashes["rollout"],
                    source_summary=str(summary_path.relative_to(output)), source_summary_sha256=hashes["summary"],
                    final_acoustic_error=summary["final"]["acoustic_error"],
                ))
    _write_json(figures / "presentation_spectrograms_metadata.json", metadata)
    return metadata


def _control(record):
    value = record.get("physical_controls", record.get("controls"))
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim == 2 and len(array):
        # Constant-control tasks should hold the same five physical values.
        if not np.all(array == array[0]):
            raise ValueError("A constant-control record contains a time-varying physical trajectory")
        array = array[0]
    return array if array.shape == (5,) else None


def _control_text(record):
    control = _control(record)
    return "—" if control is None else f"ε_l={_number(control[0])}; p_s={_number(control[2])}"


def _historical(output, manifest):
    value = manifest.get("historical_reference_directory")
    if not value:
        return [], None
    root = Path(value)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    if not root.is_relative_to((PROJECT_ROOT / "Deeplearning" / "results").resolve()):
        raise ValueError("The historical reference must be a public synthetic result below Deeplearning/results")
    rows, checkpoint_rows = [], []
    for method in METHODS:
        for seed in SEEDS:
            directory = root / "runs" / "A" / method / f"seed_{seed}"
            path = directory / "summary.json"
            if not path.exists():
                raise FileNotFoundError(path)
            summary = _read(path)
            curve_path = directory / "curve.json"
            curve = _read(curve_path)
            best = min(curve, key=lambda row: row["acoustic_error"])
            current_path = Path(output) / "runs" / "constant_1d" / method / f"seed_{seed}" / "curve.json"
            current_curve = _read(current_path)
            current_points = {point["episodes"]: point for point in current_curve}
            shared = []
            missing = []
            for point in curve:
                trial = point["episodes"]
                current = current_points.get(trial)
                if current is None:
                    missing.append(trial)
                    continue
                old_error = float(point["acoustic_error"])
                current_error = float(current["acoustic_error"])
                delta = abs(current_error - old_error)
                match = current_error == old_error
                shared.append((delta, match))
                checkpoint_rows.append(dict(method=method, seed=seed, episodes=trial,
                                            historical_acoustic_error=old_error,
                                            current_1d_acoustic_error=current_error,
                                            exact_match=match, absolute_delta=delta))
            rows.append(dict(method=method, seed=seed,
                             initial_acoustic_error=summary["initial_metrics"]["acoustic_error"],
                             final_acoustic_error=summary["final_metrics"]["acoustic_error"],
                             best_observed_acoustic_error=best["acoustic_error"],
                             best_observed_episodes=best["episodes"], evaluation_points=len(curve),
                             training_episodes=summary["training"]["episodes"],
                             training_integration_s=summary["training"]["integration_s"],
                             evaluation_episodes=summary["evaluation"]["episodes"],
                             evaluation_integration_s=summary["evaluation"]["integration_s"],
                             current_curve_points=len(current_curve),
                             shared_points_checked=len(shared),
                             shared_points_exact_matches=sum(match for _, match in shared),
                             all_historical_points_present=not missing,
                             shared_errors_exactly_equal=not missing and all(match for _, match in shared),
                             max_abs_acoustic_error_delta=max((delta for delta, _ in shared), default=None),
                             missing_historical_episodes=",".join(str(trial) for trial in missing),
                             current_curve_sha256=_sha(current_path),
                             source_summary=str(path.relative_to(PROJECT_ROOT)),
                             source_summary_sha256=_sha(path), source_curve_sha256=_sha(curve_path)))
    _csv(Path(output) / "historical_A_shared_checkpoints.csv",
         ["method", "seed", "episodes", "historical_acoustic_error", "current_1d_acoustic_error",
          "exact_match", "absolute_delta"], checkpoint_rows)
    return rows, root


def report(output):
    """Write the complete 24-run report, numeric CSVs and two new seed-0 plots."""
    output = Path(output)
    manifest = _read(output / "manifest.json")
    runs = _load_runs(output)
    if len(runs) != len(CONDITIONS) * len(METHODS) * len(SEEDS):
        raise ValueError("The complete report requires 2 conditions x 4 methods x 3 seeds")
    for run in runs:
        _curve_values(run)
    plot_metadata = plot_learning_curves(output)
    presentation_spectrograms(output)
    historical, historical_root = _historical(output, manifest)
    groups = defaultdict(list)
    per_seed, resources = [], []
    for run in runs:
        groups[(run["condition"], run["method"])].append(run)
        row = {key: run[key] for key in ("condition", "method", "seed")}
        for stage in ("initial", "final", "best_observed"):
            record = run[stage]
            control = _control(record)
            row[f"{stage}_acoustic_error"] = record["acoustic_error"]
            row[f"{stage}_episodes"] = record.get("episodes")
            row[f"{stage}_epsilon_l"] = None if control is None else float(control[0])
            row[f"{stage}_ps"] = None if control is None else float(control[2])
        row["final_minus_best"] = run["final"]["acoustic_error"] - run["best_observed"]["acoustic_error"]
        row["evaluation_points"] = len(run["_curve"])
        per_seed.append(row)
        for stage in ("training", "evaluation"):
            resource = {key: run[key] for key in ("condition", "method", "seed")}
            resource["stage"] = stage
            resource.update({key: run.get(stage, {}).get(key) for key in ACCOUNT_FIELDS})
            resources.append(resource)
    _csv(output / "per_seed.csv", list(per_seed[0]), per_seed)
    _csv(output / "compute.csv", ["condition", "method", "seed", "stage", *ACCOUNT_FIELDS], resources)
    seed0 = [row for row in per_seed if row["seed"] == 0]
    _csv(output / "seed0_diagnosis.csv", list(per_seed[0]), seed0)
    aggregates = []
    for (condition, method), selected in groups.items():
        row = dict(condition=condition, method=method, seeds=",".join(str(run["seed"]) for run in selected))
        for stage in ("initial", "final", "best_observed"):
            mean, std = _mean_std([run[stage]["acoustic_error"] for run in selected])
            row[f"{stage}_mean"], row[f"{stage}_sample_std"] = mean, std
        aggregates.append(row)
    _csv(output / "aggregate.csv", list(aggregates[0]), aggregates)
    if historical:
        _csv(output / "historical_A_reference.csv", list(historical[0]), historical)

    lines = [
        "# Issue #13：一定制御の1変数・2変数比較", "",
        "モデルCase Aの同じ0.35秒のモデル内生成目標に対し、一定1変数と一定2変数の制御を"
        "TD3・SAC・ランダム探索・差分進化で比較した。各条件・手法・seedで256本の訓練候補音を生成し、"
        "seed 0/1/2の計24 runを新しいフォルダに保存した。実録音は使用せず、既存結果は変更していない。", "",
        "## 条件と評価", "",
        _table(["条件名", "可動値", "1試行の決定", "候補音", "訓練予算/seed"], [
            ["constant_1d", "ε_l", "1回決めて全区間保持", "0.35秒", "256 episode / 候補"],
            ["constant_2d", "ε_l と p_s", "2値を1回決めて全区間保持", "0.35秒", "256 episode / 候補"],
        ]), "",
        "両条件は1 episode・1行動の一定制御課題で、35個の時変行動を学ぶ課題とは異なる。"
        "モデルCase A/B/Cという音源条件とも別の分類で、今回は全runがCase Aである。"
        "非学習変数は目標生成時の値へ固定し、初期状態・遅延履歴・出力フィルタを揃える。"
        "2変数化は左の係数と共通圧力を動かすことであり、左右2音源の比較ではない。", "",
        "主評価は既存 `acoustics.distance()` の全0.35秒の対数PSD誤差＋5 ms RMS包絡誤差。"
        "小さいほどよく、生成音ごとのゲイン・位相調整はしない。γ=1の報酬和はこの全区間誤差の負値に一致する。"
        "SACの訓練にはentropy項もあるため、全手法の学習目的の全てが同一とは述べない。"
        "balanced等の別診断値を主指標と混同しない。物理範囲・目標・学習条件の完全記録は "
        "[manifest.json](manifest.json)。", "",
        "## seed 0の学習曲線", "",
        "次の2図は**seed 0のみ**の4本の曲線で、seed平均・標準偏差ではない。"
        "横軸は `Number of trials`、縦軸は `Acoustic error`。SciencePlotsの字体・軸ラベル46 pt・"
        "目盛と凡例30 pt・線幅2.8を使い、両図の縦軸範囲を共通にした。"
        "凡例はグラフ内右上の余白に置き、データ線との非重なりを確認した。"
        "タイトル、seed表記、0.08線、帯は画像に入れていない。"
        "原点の目盛文字が重ならないよう下端に共通の余白を設けるが、実測誤差はすべて非負である。", "",
        "![constant 1d seed 0 learning curves](figures/constant_1d_learning_curves.png)", "",
        "図1：一定1変数・seed 0。", "",
        "![constant 2d seed 0 learning curves](figures/constant_2d_learning_curves.png)", "",
        "図2：一定2変数・seed 0。", "",
        "trial 0は学習・探索前の独立評価。RLでは未学習の決定論的actor、直接法では初期ランダム候補であり、"
        "同じ初期音ではない。trial 1〜256は生成済みの訓練episode／候補数を表す。"
        "RLは各時点の決定論的actor、直接法はその時点までの最良候補（best-so-far）を描くため、"
        "直接法の曲線は単調でもRLの曲線は単調でない場合がある。"
        "直接法の各試行そのものの値は `candidate_error` としてJSONに残しており、図の線はそれとは異なる。", "",
        "RLの中間episode 1〜255の評価は `_on_rollout_end` のgradient更新前、最後の256は最終更新後。"
        "各recordの `evaluation_phase`・`gradient_updates` を参照する。"
        "図はJSONの257点をそのまま結び、補間・平滑化・点の間引きはしない。"
        "Matplotlibの `path.simplify=False` とし、描画時の頂点省略も無効にする。"
        "描画した座標は [seed0_learning_curve_points.csv](seed0_learning_curve_points.csv)、"
        "軸・書式・入力ハッシュは [plot_metadata.json](figures/plot_metadata.json)。", "",
        "## 最終性能と最良観測", "",
        "次表は全3 seedの平均±標本標準偏差。最終はRLの最終actor、直接法は返された最良候補である。"
        "最良観測は257評価点の最小値で、最終性能の維持とは別に扱う。"
        "初期・最終・最良の全seed値は [per_seed.csv](per_seed.csv)、集計は [aggregate.csv](aggregate.csv)。", "",
        _table(["条件", "手法", "初期誤差", "最終誤差", "最良観測誤差"], [
            [condition, LABELS[method], *[_mean_std_text([run[stage]["acoustic_error"] for run in selected])
                                        for stage in ("initial", "final", "best_observed")]]
            for (condition, method), selected in groups.items()
        ]), "",
        "## seed 0：初期・最終・最良の差", "",
        _table(["条件", "手法", "初期", "最良観測", "最良時trial", "最終", "最終−最良", "最終物理制御"], [
            [run["condition"], LABELS[run["method"]], _number(run["initial"]["acoustic_error"]),
             _number(run["best_observed"]["acoustic_error"]), run["best_observed"].get("episodes", "—"),
             _number(run["final"]["acoustic_error"]),
             _number(run["final"]["acoustic_error"] - run["best_observed"]["acoustic_error"]),
             _control_text(run["final"])] for run in runs if run["seed"] == 0
        ]), "",
    ]
    for condition in CONDITIONS:
        chosen = [run for run in runs if run["condition"] == condition]
        best = min(chosen, key=lambda run: run["final"]["acoustic_error"])
        lines += [f"{condition}の最小最終誤差は {_number(best['final']['acoustic_error'])} "
                  f"（{best['method']}、seed {best['seed']}）。"
                  "単一seedの最小値だけで手法の優位性を判断せず、上表の3 seedのばらつきと併せて読む。", ""]
    lines += [
        "最良観測の値と、その時点の制御・音声・方策の保存は区別する。"
        "今回のRLは `best_observed/agent.zip` と `best_observed/rollout.npz`、"
        "初期は `initial/rollout.npz`、最終はrun直下の `agent.zip`・`rollout.npz` に対応する。"
        "直接法は方策ネットワークを学習しないため、生成制御列と目的値を参照する。"
        "対応する初期・最良・最終recordと保存artifactの評価時点を確認し、最終図を途中最良の図として扱わない。", "",
        "## 旧Phase 1・Task Aとの比較", "",
    ]
    if historical:
        lines += [
            f"参照先は `{historical_root.relative_to(PROJECT_ROOT)}`。旧Task Aも一定1変数の256 episode課題だが、"
            "以下は既存記録との補助比較であり、旧実験の再描画・追加学習はしていない。"
            "とくに旧曲線9点と新曲線257点では最良値を観測する密度が違う。"
            "新しい最良観測値が小さい場合も、記録点数の増加だけでは学習法の改善を証明できない。"
            "最良方策の保存を追加したことと、学習の安定性が改善したことも区別する。", "",
            _table(["手法", "seed", "旧A最終", "新1変数最終", "旧A最良観測", "新1変数最良観測", "旧/新曲線点数", "旧時点すべて完全一致", "共通時点の最大絶対差"], [
                [old["method"], old["seed"], _number(old["final_acoustic_error"]),
                 _number(next(r for r in runs if r["condition"] == "constant_1d" and r["method"] == old["method"] and r["seed"] == old["seed"])["final"]["acoustic_error"]),
                 _number(old["best_observed_acoustic_error"]),
                 _number(next(r for r in runs if r["condition"] == "constant_1d" and r["method"] == old["method"] and r["seed"] == old["seed"])["best_observed"]["acoustic_error"]),
                 f"{old['evaluation_points']}/{old['current_curve_points']}",
                 old["shared_errors_exactly_equal"], _number(old["max_abs_acoustic_error_delta"])] for old in historical
            ]), "",
            f"旧評価時点の全値と完全一致したrunは {sum(row['shared_errors_exactly_equal'] for row in historical)}/{len(historical)}。"
            f"照合した共通時点は合計 {sum(row['shared_points_checked'] for row in historical)} 点で、"
            f"うち {sum(row['shared_points_exact_matches'] for row in historical)} 点が浮動小数点値として完全一致した。"
            "完全一致の判定は許容誤差による近似比較ではない。各時点の旧値・新値・絶対差は "
            "[historical_A_shared_checkpoints.csv](historical_A_shared_checkpoints.csv)。"
            "共通時点が一致しても、新しい密な評価で見つかった途中最良を学習条件の改善とは扱わない。", "",
            "元の初期値・訓練/評価費用・出典ハッシュは [historical_A_reference.csv](historical_A_reference.csv)。"
            "新旧のハイパーパラメータ等の一致はmanifestの記録を確認する。"
            "一定2変数は旧Aから可動自由度を増やした別課題で、旧Aの単純な再実行とは扱わない。", "",
        ]
    else:
        lines += ["manifestに `historical_reference_directory` がないため、旧Aの数値比較は行っていない。", ""]
    lines += [
        "## 計算量と評価・選択の費用", "",
        "訓練episode／候補は各256本で、0.35秒×256=89.6秒のモデル時間を積分する。"
        "1試行1行動のため訓練decision stepも256。RLは初期＋各episode後の257回を独立に評価し、"
        "評価積分は89.95秒/runとなる。直接法の曲線は訓練で算出した目的値を再利用し、"
        "初期と最終だけを独立に評価するため、評価2本・0.7秒/runである。"
        "曲線点数が同じでも独立再生の本数は同じではない。"
        "最良方策の保存処理が学習器の行動・重み更新にフィードバックする実験とは区別する。", "",
        _table(["条件", "手法", "seed", "訓練行動", "訓練episode", "訓練積分[s]", "評価episode", "評価積分[s]", "訓練実時間[s]", "評価実時間[s]", "failure"], [
            [run["condition"], run["method"], run["seed"],
             *[_number(run["training"].get(key)) for key in ("decision_steps", "episodes", "integration_s")],
             *[_number(run["evaluation"].get(key)) for key in ("episodes", "integration_s")],
             _number(run["training"].get("wall_s")), _number(run["evaluation"].get("wall_s")),
             _number(run["training"].get("failures", 0) + run["evaluation"].get("failures", 0))]
            for run in runs
        ]), "",
        "会計の全値は [compute.csv](compute.csv)。native_stepsは元の積分刻み数、integration_sはモデル時間、"
        "wall_sは実時間であり別の単位である。図・音声artifactの保存や準備の費用は各summary・manifestの記録を参照。"
        "実時間は共有計算機上の測定で、同時処理・損失計算・ネットワーク更新・保存処理の影響を受ける。"
        "シミュレーション予算を揃えても全計算量・実時間が等しいとは述べない。", "",
        "## 元データとartifactの索引", "",
        "runの配置は `runs/<condition>/<method>/seed_<seed>/`。条件は `constant_1d`／`constant_2d`、"
        "手法は `TD3`／`SAC`／`random_search`／`DE`、seedは0/1/2。", "",
        _table(["ファイル", "用途"], [
            ["manifest.json", "目標・物理範囲・固定値・学習条件・予算・旧結果参照先"],
            ["runs/.../curve.json", "257点の生記録、評価時点・更新回数、直接法の各候補とbest-so-far"],
            ["runs/.../summary.json", "初期・最終・最良の完全record、訓練/評価会計"],
            ["runs/.../initial/rollout.npz", "初期音・適用制御・行動"],
            ["runs/.../best_observed/rollout.npz", "最良観測時点の音・適用制御・行動"],
            ["runs/.../best_observed/agent.zip", "RL最良観測時点の方策"],
            ["runs/.../rollout.npz", "最終音・目標音・適用制御・行動、float64の正本"],
            ["runs/.../spectrogram.png", "最終のTarget/Generated比較、最良時点と区別"],
            ["figures/<condition>_<method>_final_spectrogram.png", "seed 0・trial 256の発表用比較。両段とも画像内タイトルなし"],
            ["runs/.../amplitude_envelope.png; controls.png", "元振幅の包絡と実際の一定制御"],
            ["runs/.../target_raw.wav; generated_raw.wav", "振幅を個別正規化しないIEEE float32の評価用音声"],
        ]), "",
        "具体的なseed 0の元データ：", "",
        *[f"- [{condition}/{method} curve](runs/{condition}/{method}/seed_0/curve.json)、"
          f"[summary](runs/{condition}/{method}/seed_0/summary.json)、"
          f"[最終スペクトログラム](runs/{condition}/{method}/seed_0/spectrogram.png)、"
          f"[発表用・タイトルなし最終スペクトログラム](figures/{condition}_{method}_final_spectrogram.png)、"
          f"[最終NPZ](runs/{condition}/{method}/seed_0/rollout.npz)、"
          f"[最終生成WAV](runs/{condition}/{method}/seed_0/generated_raw.wav)"
          for condition in CONDITIONS for method in METHODS], "",
        "発表用スペクトログラムのcaption：**上段が目標音（Target）、下段が最終生成音（Generated）。"
        "すべてseed 0・trial 256の最終recordで、途中最良の図ではない。**"
        "画像内・上側・枠外のタイトルを置かず、両段の周波数軸46 pt・目盛30 pt、時間軸を `Time [s]` とする。"
        "元のrun内 `spectrogram.png` は変更せず、保存済み最終NPZだけから別画像として描いた。"
        "元NPZ・summaryと各PNGの対応ハッシュは "
        "[presentation_spectrograms_metadata.json](figures/presentation_spectrograms_metadata.json)。"
        "この描画で主音響誤差の再計算や物理モデルの再積分は行っていない。", "",
        "元振幅のNPZ/WAVは保持する。既存スペクトログラムの表示専用ピーク正規化は学習・評価には適用しない。"
        "出力はモデルの単位で、SI音圧への校正や公式SAP評価を実施したことにはならない。", "",
        "## 判断の範囲と未解決事項", "",
        "一定1/2変数の単一固定目標・初期状態における実験である。最終誤差、途中最良、最良artifactの保存、"
        "その後の性能維持を分けて判断する。新しい図のseed 0だけで3 seed全体の性質を代表させない。"
        "パラメータ一致は音響的成功の必須条件ではなく、異なる制御で近い音が出る可能性を残す。"
        "結果が悪い場合も有限の予算・探索/entropy・関数近似・候補選択・観測条件の影響が残り、"
        "RL一般の不可能性やモデル表現力不足とは結論しない。", "",
        "今回の結果は時変目標、複数目標、未知の初期状態、実録音への一般化を検証していない。"
        "左右音源の使用条件を変えた実験でもなく、左右2音源の優位性を主張しない。"
        "0.08等の過去の比較線は知覚的な合格基準ではなく、今回の図には描いていない。", "",
    ]
    lines += ["## 保存結果の監査", ""]
    audit_path = output / "final_audit.json"
    if audit_path.exists():
        audit = _read(audit_path)
        passed = audit.get("passed")
        status = "PASS" if passed is True else "FAIL" if passed is False else "判定値なし"
        checks = audit.get("checks")
        check_count = len(checks) if isinstance(checks, (dict, list)) else checks
        lines += [f"最終監査：**{status}**、確認数 {_number(check_count)}。"
                  "[final_audit.json](final_audit.json) の保存値を掲載する。"
                  "このレポートの再生成は学習・数値積分を追加せず、既存の監査結果を読み込む。", ""]
        counts = audit.get("counts", {})
        if counts:
            lines += [_table(["監査項目", "件数・保存値"], [[key, _number(value)] for key, value in counts.items()]), ""]
        if audit.get("errors"):
            lines += ["監査で記録された問題：", "", *[f"- {error}" for error in audit["errors"]], ""]
    else:
        lines += ["`final_audit.json` はまだ生成されていない。監査後にレポートを再生成すると、"
                  "保存済みのPASS/FAILと確認数を反映する。", ""]
    lines += ["## 完全な条件記録", "",
              "```json", json.dumps(manifest, ensure_ascii=False, indent=2), "```", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return output / "REPORT.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--all-seeds", action="store_true",
                        help="With --plots-only, save separate learning curves for seeds 0, 1 and 2")
    args = parser.parse_args()
    if args.all_seeds:
        if not args.plots_only:
            parser.error("--all-seeds requires --plots-only")
        plot_all_seed_curves(args.output)
    elif args.plots_only:
        plot_learning_curves(args.output)
    else:
        print(report(args.output))
