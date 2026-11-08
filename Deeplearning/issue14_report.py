"""Independent plots and reports for the fixed-target Phase 0 experiment.

The experiment runner is deliberately not imported.  Evaluation values come
from JSON; figures use the corresponding saved final waveforms and controls.
No policy rollout, model integration, or acoustic-loss recomputation is done.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 registers styles

from .plotting import (CMAP, NOVERLAP, NPERSEG, PHASE0_MAX_HZ, TARGET_FS,
                       VMAX, VMIN, WINDOW_TYPE, draw_spectrogram,
                       phase0_all_seed_learning_curves)


ROOT = Path(__file__).resolve().parent
ALGORITHMS = ("PPO", "TD3", "SAC")
SEEDS = (0, 1, 2)
COLORS = {"PPO": "#0072B2", "TD3": "#009E73", "SAC": "#D55E00"}
EVALUATION_STEPS = np.arange(0, 4097, 512)
TRACKING_FIELDS = ("frequency_mae_hz", "frequency_max_abs_hz", "amplitude_mae", "amplitude_max_abs")


def _read(path):
    return json.loads(Path(path).read_text())


def _optional(path):
    return _read(path) if Path(path).exists() else None


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _number(value, digits=7):
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _meanstd(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1 if len(values) > 1 else 0))


def _meanstd_text(values):
    mean, std = _meanstd(values)
    return f"{mean:.6g} ± {std:.3g}"


def _table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |" for row in rows),
    ])


def _error(record):
    return float(record["error"] if "error" in record else record["acoustic_error"])


def _load(output):
    runs = []
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            directory = Path(output) / f"{algorithm}_{seed}"
            summary = _read(directory / "summary.json")
            if (summary["algorithm"], summary["seed"]) != (algorithm, seed):
                raise ValueError(f"{directory}: algorithm/seed mismatch")
            curve = _read(directory / "curve.json")
            callback = _read(directory / "callback_curve.json")
            steps = np.asarray([point["steps"] for point in curve])
            errors = np.asarray([_error(point) for point in curve])
            if not np.array_equal(steps, EVALUATION_STEPS):
                raise ValueError(f"{directory}: expected the nine evaluations at 0, 512, ..., 4096")
            if not np.isfinite(errors).all() or np.any(errors < 0):
                raise ValueError(f"{directory}: invalid saved evaluation errors")
            if errors[0] != summary["initial_error"] or errors[-1] != summary["error"]:
                raise ValueError(f"{directory}: curve endpoints and summary disagree")
            if curve[-1]["evaluation_phase"] != "post_final_update":
                raise ValueError(f"{directory}: the plotted endpoint must follow the final weight update")
            if curve[-1]["parameter_sha256"] != summary["final_parameter_sha256"]:
                raise ValueError(f"{directory}: final parameter fingerprints disagree")
            if any(point.get("learner_unchanged") is not True for point in curve):
                raise ValueError(f"{directory}: evaluation must leave the learner unchanged")
            if any(point.get("learner_unchanged") is not True for point in callback):
                raise ValueError(f"{directory}: callback evaluation must leave the learner unchanged")
            with np.load(directory / "rollout.npz", allow_pickle=False) as saved:
                snapshot = {key: saved[key].copy() for key in saved.files}
            controls, target_controls = snapshot["physical_controls"], snapshot["target_controls"]
            if controls.shape != (35, 2) or target_controls.shape != (35, 2):
                raise ValueError(f"{directory}: expected 35 pairs of frequency/amplitude controls")
            if (snapshot["audio"].shape != (16800,) or snapshot["target"].shape != (16800,)
                    or int(snapshot["sample_rate"].item()) != 48000):
                raise ValueError(f"{directory}: expected complete 0.35 s, 48 kHz final audio")
            if snapshot["actions"].shape != (35, 2) or snapshot["step_rewards"].shape != (35,):
                raise ValueError(f"{directory}: incomplete final actions or rewards")
            absolute = np.abs(controls - target_controls)
            tracking = dict(frequency_mae_hz=float(absolute[:, 0].mean()),
                            frequency_max_abs_hz=float(absolute[:, 0].max()),
                            amplitude_mae=float(absolute[:, 1].mean()),
                            amplitude_max_abs=float(absolute[:, 1].max()))
            if not np.allclose([tracking[key] for key in TRACKING_FIELDS],
                               [summary["tracking"][key] for key in TRACKING_FIELDS], rtol=1e-10, atol=1e-10):
                raise ValueError(f"{directory}: tracking summary and saved physical controls disagree")
            runs.append(dict(summary, _directory=directory, _curve=curve, _callback=callback,
                             _steps=steps, _errors=errors, _snapshot=snapshot,
                             _absolute_control_errors=absolute))
    return runs


def _legend_overlaps(ax, legend, series):
    box = legend.get_window_extent(ax.figure.canvas.get_renderer())
    for steps, errors in series:
        xy = ax.transData.transform(np.column_stack((steps, errors)))
        for a, b in zip(xy[:-1], xy[1:]):
            if (max(a[0], b[0]) >= box.x0 and min(a[0], b[0]) <= box.x1
                    and max(a[1], b[1]) >= box.y0 and min(a[1], b[1]) <= box.y1):
                return True
    return False


def _learning_plot(runs, path):
    selected = [run for run in runs if run["seed"] == 0]
    maximum = max(float(run["_errors"].max()) for run in selected)
    upper = max(.2, maximum * 1.5)
    with plt.style.context(["science", "ieee", "no-latex"]), plt.rc_context({"path.simplify": False}):
        fig, ax = plt.subplots(figsize=(16, 10))
        try:
            for run in selected:
                ax.plot(run["_steps"], run["_errors"], color=COLORS[run["algorithm"]],
                        linewidth=2.8, linestyle="-", alpha=.8, label=run["algorithm"])
            ax.set_xlim(0, 4096)
            ax.set_ylim(-upper * .025, upper)
            ax.set_xticks(EVALUATION_STEPS)
            ax.set_xlabel("Training steps", fontsize=46)
            ax.set_ylabel("Acoustic error", fontsize=46)
            ax.tick_params(labelsize=30, width=2, length=10, direction="in", top=False, right=False)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            legend = ax.legend(loc="upper right", ncol=3, fontsize=30, frameon=False)
            fig.tight_layout()
            fig.canvas.draw()
            if _legend_overlaps(ax, legend, [(run["_steps"], run["_errors"]) for run in selected]):
                raise RuntimeError("The main legend overlaps a measured learning curve")
            bounds = list(ax.get_ylim())
            fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=.1)
            return dict(seed=0, colors=COLORS, line_width=2.8, line_style="solid",
                        axis_fontsize=46, tick_fontsize=30, legend_fontsize=30,
                        legend="inside upper right", legend_overlaps_data=False,
                        ylim=bounds, title=None, threshold_line=False, uncertainty_band=False,
                        path_simplify=False, source_values="curve.json, without smoothing or resampling")
        finally:
            plt.close(fig)


def _spectrogram(run, path):
    snapshot = run["_snapshot"]
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        try:
            for ax, waveform in zip(axes, (snapshot["target"], snapshot["audio"])):
                draw_spectrogram(ax, waveform, 48000, max_hz=PHASE0_MAX_HZ)
                ax.set_title("")
                ax.set_ylabel("Frequency [Hz]", fontsize=46)
                ax.tick_params(labelsize=30)
            axes[-1].set_xlabel("Time [s]", fontsize=46)
            fig.tight_layout()
            fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=.1)
        finally:
            plt.close(fig)


def _control_plot(run, path):
    snapshot = run["_snapshot"]
    times = np.arange(36) * .01
    with plt.style.context(["science", "ieee", "no-latex"]), plt.rc_context({"path.simplify": False}):
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        try:
            for index, ax in enumerate(axes):
                for values, color, style, label in (
                    (snapshot["target_controls"][:, index], "black", "-", "Target"),
                    (snapshot["physical_controls"][:, index], COLORS[run["algorithm"]], "-", "Generated"),
                ):
                    ax.step(times, np.r_[values, values[-1]], where="post", linewidth=2.8,
                            color=color, linestyle=style, label=label)
                ax.set_ylabel("Frequency [Hz]" if index == 0 else "Amplitude", fontsize=46)
                ax.tick_params(labelsize=30, width=2, length=10, direction="in", top=False, right=False)
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                ax.legend(loc="upper right", ncol=2, fontsize=30, frameon=False)
            axes[0].set_ylim(500, 3000)
            axes[1].set_ylim(-.03, 1)
            axes[1].set_xlim(0, .35)
            axes[1].set_xlabel("Time [s]", fontsize=46)
            fig.tight_layout()
            fig.canvas.draw()
            for ax in axes:
                if _legend_overlaps(ax, ax.get_legend(), [(line.get_xdata(), line.get_ydata()) for line in ax.lines]):
                    raise RuntimeError("A control-plot legend overlaps a target/generated trace")
            fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=.1)
        finally:
            plt.close(fig)


def _source_metadata(output, runs):
    sources, correspondence = {}, []
    for relative in ("manifest.json", "config.json", "diagnostics/oracle.json", "diagnostics/rollout.npz"):
        path = Path(output) / relative
        if path.exists():
            sources[relative] = _sha(path)
    for run in runs:
        directory = run["_directory"]
        for name in ("summary.json", "curve.json", "callback_curve.json", "rollout.npz",
                     "initial/rollout.npz", "agent.zip", "training_trace.npz"):
            path = directory / name
            if path.exists():
                sources[str(path.relative_to(output))] = _sha(path)
        final = run["_curve"][-1]
        correspondence.append(dict(algorithm=run["algorithm"], seed=run["seed"],
                                   final_steps=final["steps"], curve_final_error=_error(final),
                                   summary_final_error=run["error"], errors_exactly_equal=True,
                                   evaluation_phase=final["evaluation_phase"],
                                   final_parameter_sha256=run["final_parameter_sha256"],
                                   final_parameter_fingerprints_equal=True,
                                   recorded_learner_signature_checks_passed=True))
    return sources, correspondence


def _oracle_text(output):
    oracle = _optional(Path(output) / "diagnostics" / "oracle.json")
    if oracle is None:
        return "Oracleの診断記録は未保存。完了確認では数値と許容誤差の記録が必要。"
    metrics = oracle.get("metrics", {})
    error = oracle.get("error", oracle.get("acoustic_error", metrics.get("total")))
    observed = f"保存値は {_number(error)}。" if error is not None else "保存された各方式の値を診断JSONで確認する。"
    return ("既知の目標f・Aを直接適用するOracleは、従来のfloat32正規化行動では丸めを含むため、"
            "完全な0とは区別する。" + observed +
            "従来精度の範囲（1e−5未満）での再現を確認し、float64の物理f・Aを直接生成する別診断と混同しない。"
            "[Oracle記録](diagnostics/oracle.json) は学習成績ではない。")


def _implementation_section(output):
    tests = _optional(Path(output) / "test_results.json") or {}
    audit = _optional(Path(output) / "audit.json") or {}
    execution = _optional(Path(output) / "execution.json") or {}
    tests_status = "PASS" if tests.get("passed") is True else "記録参照"
    audit_status = "PASS" if audit.get("passed") is True else "記録参照"
    wall = execution.get("execution_wall_s")
    wall_text = "—" if wall is None else f"{wall:.3f}"
    return [
        "## コード変更と再現・検証", "",
        "- [experiments.py](../../experiments.py) の `phase0_factories` は、学習目標を乱数化する旧default=Trueを維持し、Falseで固定目標へ切り替える。",
        "- [issue14.py](../../issue14.py) は固定目標専用の保存・評価pipeline。更新前callbackと最終更新後の評価を分け、発表用曲線の終点を後者へ揃える。"
        "[新config](../../configs/issue14_fixed_target.json) に条件を固定し、実効値・版・ハッシュをmanifestと各summaryへ保存した。",
        "- [issue14_audit.py](../../issue14_audit.py) は保存音声・方策・値の対応を監査し、"
        "[test_issue14.py](../../tests/test_issue14.py) は固定目標・旧default・評価時の更新禁止等を検証する。"
        "[issue14_report.py](../../issue14_report.py) は保存値だけから本レポート・図・CSVを生成する。", "",
        f"関連テストは {_number(tests.get('tests_passed'))} 件 **{tests_status}** "
        f"（[test_results.json](test_results.json)）、独立監査は {_number(audit.get('checks'))} 確認 **{audit_status}** "
        "（[audit.json](audit.json)）。回帰テストの一時モデルと、本実験9runの訓練予算を分けて記録している。", "",
        f"9runの実行（学習・run内評価・保存等）は {wall_text} 秒。訓練 "
        f"{_number(execution.get('training_steps'))} stepと評価 {_number(execution.get('evaluation_steps'))} stepを分けて計上した "
        "（[execution.json](execution.json)）。各runの評価実時間は前表に記録し、Oracle準備・独立監査・図の生成は"
        "このexecution実時間とは別の費用である。", "",
        "学習を再実行する場合はmanifestの実行コマンドを参照し、**新しい実験ID・未使用の出力フォルダ**を指定する。"
        "既存保存先は上書きしない。今回の報告作成では再学習せず、既存JSON/NPZの読み取りで図と文書を生成した。", "",
    ]


def generate_report(output: Path):
    """Read nine completed fixed-target runs; write only new report artifacts."""
    started = time.perf_counter()
    output = Path(output)
    manifest = _read(output / "manifest.json")
    runs = _load(output)
    sources, correspondence = _source_metadata(output, runs)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_settings = _learning_plot(runs, figures / "learning_curves_seed0.png")
    figure_records = [dict(path="figures/learning_curves_seed0.png", stage="plotted evaluations; final endpoint post-update",
                           source_curves=[f"{name}_0/curve.json" for name in ALGORITHMS], **plot_settings)]
    all_curves = {name: {run["seed"]: (run["_steps"], run["_errors"])
                        for run in runs if run["algorithm"] == name} for name in ALGORITHMS}
    phase0_all_seed_learning_curves(all_curves, figures / "learning_curves_all_seeds.png")
    figure_records.append(dict(path="figures/learning_curves_all_seeds.png", role="supplementary",
                               marker_size_pt=14, markers={0: "o", 1: "s", 2: "^"},
                               source_curves=[f"{name}_{seed}/curve.json" for name in ALGORITHMS for seed in SEEDS]))
    for run in runs:
        if run["seed"] != 0:
            continue
        name = run["algorithm"]
        for kind, render in (("final_spectrogram", _spectrogram), ("controls", _control_plot)):
            filename = f"{name}_seed0_{kind}.png"
            render(run, figures / filename)
            figure_records.append(dict(path=f"figures/{filename}", algorithm=name, seed=0,
                                       stage="final", training_steps=4096, titles=[],
                                       control_line_style="solid" if kind == "controls" else None,
                                       control_legend="inside upper right of each panel" if kind == "controls" else None,
                                       source_rollout=f"{name}_0/rollout.npz", source_summary=f"{name}_0/summary.json",
                                       final_parameter_sha256=run["final_parameter_sha256"]))
    for record in figure_records:
        record["png_sha256"] = _sha(output / record["path"])

    per_seed, control_rows, worst, old_new, aggregates = [], [], [], [], []
    old_sources = {}
    grouped = defaultdict(list)
    for run in runs:
        name, seed = run["algorithm"], run["seed"]
        grouped[name].append(run)
        row = dict(algorithm=name, seed=seed, initial_error=run["initial_error"], final_error=run["error"],
                   initial_log_spectral=run["initial_metrics"]["log_spectral"],
                   initial_envelope=run["initial_metrics"]["envelope"],
                   final_log_spectral=run["metrics"]["log_spectral"], final_envelope=run["metrics"]["envelope"],
                   **{field: run["tracking"][field] for field in TRACKING_FIELDS})
        for field in ("training_steps", "training_episodes", "training_partial_episode_steps",
                      "evaluation_steps", "evaluation_episodes", "training_generated_s", "evaluation_generated_s",
                      "training_wall_s", "evaluation_wall_s", "total_wall_s", "final_evaluation_phase"):
            row[field] = run[field]
        row["negative_sum_local_rewards"] = -float(np.sum(run["_snapshot"]["step_rewards"]))
        row["final_parameter_sha256"] = run["final_parameter_sha256"]
        per_seed.append(row)
        snapshot = run["_snapshot"]
        for index, (actual, target) in enumerate(zip(snapshot["physical_controls"], snapshot["target_controls"])):
            control_rows.append(dict(algorithm=name, seed=seed, block=index, start_s=index * .01,
                                     end_s=(index + 1) * .01, target_frequency_hz=float(target[0]),
                                     generated_frequency_hz=float(actual[0]), frequency_error_hz=float(actual[0] - target[0]),
                                     target_amplitude=float(target[1]), generated_amplitude=float(actual[1]),
                                     amplitude_error=float(actual[1] - target[1])))
        if seed == 0:
            f_block = int(np.argmax(run["_absolute_control_errors"][:, 0]))
            a_block = int(np.argmax(run["_absolute_control_errors"][:, 1]))
            worst.append(dict(algorithm=name, frequency_block=f_block,
                              frequency_interval=f"{f_block * .01:.2f}–{(f_block + 1) * .01:.2f}",
                              frequency_max_abs_hz=run["tracking"]["frequency_max_abs_hz"],
                              amplitude_block=a_block,
                              amplitude_interval=f"{a_block * .01:.2f}–{(a_block + 1) * .01:.2f}",
                              amplitude_max_abs=run["tracking"]["amplitude_max_abs"]))
        old_path = ROOT / "results" / "phase0" / f"{name}_{seed}" / "summary.json"
        old = _read(old_path)
        old_sources[str(old_path.relative_to(ROOT))] = _sha(old_path)
        old_new.append(dict(algorithm=name, seed=seed, old_initial_error=old["initial_error"],
                            new_initial_error=run["initial_error"], old_final_error=old["error"],
                            new_final_error=run["error"], new_minus_old_final=run["error"] - old["error"],
                            old_source=str(old_path.relative_to(ROOT)), old_source_sha256=_sha(old_path)))
    for name, selected in grouped.items():
        aggregate = dict(algorithm=name, seeds="0,1,2")
        for key, values in (
            ("initial_error", [run["initial_error"] for run in selected]),
            ("final_error", [run["error"] for run in selected]),
            *( (field, [run["tracking"][field] for run in selected]) for field in TRACKING_FIELDS),
        ):
            aggregate[key + "_mean"], aggregate[key + "_sample_std"] = _meanstd(values)
        aggregates.append(aggregate)
    _csv(output / "per_seed.csv", per_seed)
    _csv(output / "aggregate.csv", aggregates)
    _csv(output / "old_vs_fixed_target.csv", old_new)
    _csv(output / "control_tracking.csv", control_rows)
    _csv(output / "seed0_worst_tracking_intervals.csv", worst)

    figure_metadata = dict(
        primary_acoustic_error_recomputed=False, added_training_steps=0, added_generated_audio_s=0,
        source_sha256=sources, old_summary_sha256=old_sources, final_correspondence=correspondence,
        learner_signature_scope_limits=["replay buffer content is not fully hashed; pos/full/size are recorded",
                                        "SAC entropy coefficient outside policy is not itself included in the policy fingerprint"],
        spectrogram=dict(max_hz=PHASE0_MAX_HZ, target_fs=TARGET_FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                         window=WINDOW_TYPE, cmap=CMAP, vmin=VMIN, vmax=VMAX,
                         complete_windows_only=True, top_panel="Target", bottom_panel="Generated",
                         per_waveform_peak_normalization="display only"),
        figures=figure_records, generation_wall_s=time.perf_counter() - started,
    )
    for relative, digest in sources.items():
        if _sha(output / relative) != digest:
            raise RuntimeError(f"A report source changed while rendering: {relative}")
    _write_json(output / "figure_metadata.json", figure_metadata)

    improved = sum(run["error"] < run["initial_error"] for run in runs)
    lines = [
        "# Issue #14：固定目標音でのPhase 0再実験", "",
        "**条件：** 学習・評価とも同じ0.35秒の正弦波グライド。PPO・TD3・SAC、seed 0/1/2、"
        "各4096訓練環境stepで実行した。変更の中心は学習目標のランダム化を無効にすることで、"
        "鳴管モデル・実録音は対象に含めない。目標の周波数・振幅を観測に与える単純な追従課題である。", "",
        f"**結果の事実：** {improved}/9 runで最終の全区間音響誤差が学習前より小さかった。"
        "手法別の初期→最終平均は " + "、".join(
            f"{name} {_number(np.mean([run['initial_error'] for run in selected]))} → "
            f"{_number(np.mean([run['error'] for run in selected]))}" for name, selected in grouped.items()) + "。"
        f"全9 runの周波数MAEは {_number(min(run['tracking']['frequency_mae_hz'] for run in runs))}〜"
        f"{_number(max(run['tracking']['frequency_mae_hz'] for run in runs))} Hz、振幅MAEは "
        f"{_number(min(run['tracking']['amplitude_mae'] for run in runs))}〜"
        f"{_number(max(run['tracking']['amplitude_mae'] for run in runs))}。"
        "音響誤差の減少を完全な追従や聴覚的同一性と同一視しない。", "",
        "**旧条件との比較：** 最終音響誤差の3 seed平均（旧→固定目標）は " + "、".join(
            f"{name} {_number(np.mean([row['old_final_error'] for row in old_new if row['algorithm'] == name]))} → "
            f"{_number(np.mean([row['new_final_error'] for row in old_new if row['algorithm'] == name]))}"
            for name in ALGORITHMS) + "。差の原因や順位の一般性は断定せず、条件・乱数の制約と併せて読む。", "",
        "**まず見る5図：** すべてseed 0。学習曲線の終点とスペクトログラム・制御は最終更新後の同じ方策に対応する。", "",
        "1. [学習曲線：PPO・TD3・SAC](figures/learning_curves_seed0.png)",
        "2. [PPO：最終の目標／生成スペクトログラム](figures/PPO_seed0_final_spectrogram.png)",
        "3. [TD3：最終の目標／生成スペクトログラム](figures/TD3_seed0_final_spectrogram.png)",
        "4. [SAC：最終の目標／生成スペクトログラム](figures/SAC_seed0_final_spectrogram.png)",
        "5. [TD3：目標と生成の周波数・振幅制御](figures/TD3_seed0_controls.png)", "",
        "発表用の説明：**「鳴管モデルへの適用前に、固定目標の正弦波モデルでPPO・TD3・SACの学習挙動を確認した」。**"
        "学習にも使う同じ目標の成績であり、held-out評価や未知音への一般化とは呼ばない。", "",
        "## 条件と評価の区別", "",
        "目標周波数は35区間で1000→2000 Hz。目標振幅は既存評価の時変系列をそのまま用い、一定化しない。"
        "音声48 kHz、10 msごと35行動、初期位相はresetし、区間間では位相を連続に保つ。"
        "行動は周波数500–3000 Hzと振幅0–1、観測は時刻・その区間の目標周波数/振幅・直前行動である。"
        "学習率・gamma・network・バッファ等の実効値は各runのsummary、固定設定と版・コードハッシュはmanifestを参照。", "",
        "訓練は各10 ms区間の音響誤差E_kに対する `r_k=−E_k`。平滑化ペナルティは0。"
        "評価では別環境をresetし、探索ノイズなしの最終方策で35区間を最初から生成して、"
        "全0.35秒の音響誤差を評価する。**35個の局所報酬の和と全区間誤差は同一ではない。**"
        "主指標は従来の対数スペクトル誤差（80 dBで正規化）＋5 ms移動RMS包絡誤差で、"
        "共有スケール・微小値・窓は変更せず、表示用ピーク正規化と分ける。", "",
        "4096はepisode数ではなく制御行動数で、117完了episode＋次episodeの1step。"
        "評価は初期・512stepごとのcallback・learn終了後に計10 episode=350環境stepを別に実施する。"
        "方策更新のタイミングは手法ごとに異なる。評価前後の署名が一致する記録と、"
        "評価中のtrain/addを禁止するテストを保存した。署名はpolicy・optimizer・RNG・更新数等と、"
        "バッファのpos/full/sizeを確認するもので、バッファの全内容のハッシュではない。"
        "SACのpolicy外にあるentropy係数本体もpolicy fingerprintには含まれないため、"
        "全内部状態の完全な同一性を署名だけで証明したとは述べない。", "",
        "## 学習曲線と最終方策の対応", "",
        "![seed 0 learning curves](figures/learning_curves_seed0.png)", "",
        "図1：seed 0の各手法1本。横軸は訓練step、縦軸は全区間音響誤差。"
        "記録された9点（0,512,…,4096）をそのまま結び、平滑化・間引きはしない。"
        "最終点4096は最終更新完了後の評価へ置き換えた値で、summary・保存方策・最終音・制御に対応する。"
        "置換前の更新前callback値は `callback_curve.json` に残す。"
        "曲線9点と独立評価10本を混同せず、途中の最良だけを結果に採用しない。", "",
        _table(["手法・seed 0", "4096 callback時の更新数", "最終更新後の更新数", "callback誤差", "最終誤差"], [
            [run["algorithm"], run["_callback"][-1]["gradient_updates"], run["_curve"][-1]["gradient_updates"],
             _number(_error(run["_callback"][-1])), _number(run["error"])]
            for run in runs if run["seed"] == 0
        ]), "",
        "## 全3 seedの数値", "",
        "平均±標本標準偏差。Hzは周波数の制御追従誤差、振幅はモデルの無次元制御値。", "",
        _table(["手法", "初期音響誤差", "最終音響誤差", "周波数MAE[Hz]", "振幅MAE"], [
            [name, _meanstd_text([run["initial_error"] for run in selected]),
             _meanstd_text([run["error"] for run in selected]),
             _meanstd_text([run["tracking"]["frequency_mae_hz"] for run in selected]),
             _meanstd_text([run["tracking"]["amplitude_mae"] for run in selected])]
            for name, selected in grouped.items()
        ]), "",
        _table(["手法", "seed", "初期誤差", "最終誤差", "最終スペクトル項", "最終包絡項"], [
            [run["algorithm"], run["seed"], _number(run["initial_error"]), _number(run["error"]),
             _number(run["metrics"]["log_spectral"]), _number(run["metrics"]["envelope"])] for run in runs
        ]), "",
        "個別の追従最大誤差・実時間等も含む [per_seed.csv](per_seed.csv)、"
        "[aggregate.csv](aggregate.csv) に全値を保存した。seedは同じ固定目標での初期化・探索の乱数差であり、"
        "目標音の多様性を表す標本ではない。0.08は聴覚的同一性の合格基準として扱わない。", "",
        "## 目標と生成のスペクトログラム・制御", "",
        "スペクトログラムは**上段Target／下段Generated**、すべてseed 0・最終4096step更新後。"
        "画像に上部タイトル・seed表記を加えず、従来のPhase 0表示帯域0–2500 Hzを維持した。"
        "表示のみ10 MHzへ再サンプリングし、Blackman–Harris、nperseg=245760、noverlap=184320、"
        "gray_r、−50〜−20 dB、窓全体が含まれる中心時刻を使用する。"
        "表示用ピーク正規化の濃淡から振幅一致を判断せず、f/Aの重ね描きと追従数値を確認する。", "",
        *[f"- {name}：[最終スペクトログラム](figures/{name}_seed0_final_spectrogram.png)、"
          f"[目標／生成のf・A制御](figures/{name}_seed0_controls.png)、"
          f"[最終音・制御NPZ]({name}_0/rollout.npz)" for name in ALGORITHMS], "",
        "seed 0で最大の制御ずれが残る区間を示す。区間は10 msの操作区間で、"
        "最悪1区間だけを全体性能の代わりには扱わない。同じ最大値がある場合は最初の区間を示す。", "",
        _table(["手法", "周波数最大誤差[Hz]", "その区間[s]", "振幅最大誤差", "その区間[s]"], [
            [row["algorithm"], _number(row["frequency_max_abs_hz"]), row["frequency_interval"],
             _number(row["amplitude_max_abs"]), row["amplitude_interval"]] for row in worst
        ]), "",
        "全seed・全35区間の値は [control_tracking.csv](control_tracking.csv)。"
        "制御図は各値をその10 ms区間に保持して描き、補間・平滑化は追加していない。", "",
        "## 旧ランダム目標学習との比較", "",
        "旧条件はepisodeごとに学習グライドを乱数化し、固定1000→2000 Hz音で評価していた。"
        "新条件はその固定評価波形を学習にも使用する。旧最終値は保存済み各runのsummaryの `error` "
        "（最終更新後）から読み、旧学習曲線の4096 callback値を代用しない。旧成果は再生成していない。", "",
        _table(["手法", "旧初期誤差", "新初期誤差", "旧最終誤差", "新最終誤差"], [
            [name, _meanstd_text([row["old_initial_error"] for row in old_new if row["algorithm"] == name]),
             _meanstd_text([row["new_initial_error"] for row in old_new if row["algorithm"] == name]),
             _meanstd_text([row["old_final_error"] for row in old_new if row["algorithm"] == name]),
             _meanstd_text([row["new_final_error"] for row in old_new if row["algorithm"] == name])]
            for name in ALGORITHMS
        ]), "",
        "全seedの比較値・新旧差・旧summaryの出典ハッシュは "
        "[old_vs_fixed_target.csv](old_vs_fixed_target.csv)。旧結果もcallback末尾と最終summaryが異なる場合がある。"
        "今回の図は最終点をsummaryと一致させたため、旧PNGの末尾と新PNGの末尾だけを比べて結論しない。"
        "条件変更の前後で得た事実と原因の説明を分け、固定しただけで必ず改善する、学習不足、"
        "モデルが再現不能といった原因を今回の結果だけで断定しない。", "",
        "**乱数再現性の制約：** 旧環境の `reset(seed=None)` が行動空間をOS由来の乱数で再seedする挙動も維持した。"
        "このためseed番号だけではTD3/SACのwarm-up行動列の完全再現を保証しない。"
        "各runの `training_trace.npz` は4096stepの要求/生行動と報酬・doneの記録である。"
        "PPOの生行動は環境へ渡す前の[-1,1]へのclipより前で、環境ではfloat32化・clipを行う。"
        "TD3等でも環境のfloat32化より前の値の場合があり、適用後の物理f・Aと同一とは扱わない。"
        "manifest末尾のactual training actionsという記述も、このraw/requested記録を指す。"
        "この制約を隠して、旧条件との数値差の全てを目標固定の因果効果と断定しない。"
        "詳細はmanifestの `reproducibility_limit`。", "",
        "## 計算時間と検証", "",
        _table(["手法", "seed", "訓練step", "評価step", "訓練実時間[s]", "評価実時間[s]", "総実時間[s]"], [
            [run["algorithm"], run["seed"], run["training_steps"], run["evaluation_steps"],
             _number(run["training_wall_s"]), _number(run["evaluation_wall_s"]), _number(run["total_wall_s"])]
            for run in runs
        ]), "",
        "訓練で生成した音声長は40.96秒/run、評価は3.5秒/run。これはwall-clock時間ではなく生成音の長さの合計。"
        "描画は訓練・評価とは別の処理で、共有計算機上の実時間を厳密な速度順位として扱わない。", "",
        _oracle_text(output), "",
    ]
    audit = _optional(output / "audit.json")
    preservation = _optional(output / "preservation_check.json")
    for label, path, evidence in (("保存結果監査", "audit.json", audit),
                                  ("既存成果保護", "preservation_check.json", preservation)):
        if evidence:
            passed = evidence.get("passed")
            status = "PASS" if passed is True else "FAIL" if passed is False else "記録参照"
            lines += [f"{label}：**{status}**。詳細は [{path}]({path})。", ""]
        else:
            lines += [f"{label}の `{path}` は未生成。監査後にレポートを再生成すると結果を反映する。", ""]
    lines += _implementation_section(output)
    lines += [
        "## 出典と発表時の限界", "",
        "図と最終音・制御・重みの対応、JSON/NPZ/agentのSHA256は "
        "[figure_metadata.json](figure_metadata.json)。"
        "この報告処理は保存値と制御列を読み、学習・新しい音声生成・主音響誤差の再計算は行わない。"
        "保存音声からの主誤差の独立再計算とモデル重みの対応確認はaudit側の記録を参照する。", "",
        "補足：[全seedの学習曲線](figures/learning_curves_all_seeds.png)。色が手法、○/□/△がseed 0/1/2、"
        "14 ptの大きなマーカーと2段の凡例で区別する。主図のseed 0の3本とは別図として使用する。", "",
        "本結果は目標のf・Aを観測できる固定正弦波追従の検証で、未知の物理制御を音声だけから推定する課題ではない。"
        "鳴管への有効性、RLの必然性、左右2音源の効果、未知音への一般化をこの実験からは主張しない。"
        "成功するまで予算・係数を追加変更する実験とも区別する。", "",
        "設定・コード/ライブラリ版・実行条件は [manifest.json](manifest.json)、"
        "各runの `summary.json` の `effective_hyperparameters` を参照。完全条件はこの機械可読記録を正本とし、"
        "本レポートへのJSON全文の重複掲載は省略する。", "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    return output / "REPORT.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(generate_report(args.output))
