"""Artifacts and an evidence-based report for the synthetic Phase 1 validation.

This module reads only the result directory passed by its caller.  Original
waveform amplitudes are retained; the legacy spectrogram's display-only peak
normalization must not be mistaken for a calibrated amplitude comparison.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from .plotting import comparison, control_trajectories


METHOD_ORDER = {"TD3": 0, "SAC": 1, "random_search": 2, "DE": 3}
METHOD_LABELS = {"TD3": "TD3", "SAC": "SAC", "random_search": "Random search", "DE": "Differential evolution"}
TASK_LABELS = {"A": "1 variable, constant", "B": "1 variable, 10 ms", "C": "2 variables, 10 ms"}
METRIC_NAMES = ("acoustic_error", "balanced_error", "parameter_nrmse")


def _json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def _number(value, digits=6):
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).replace("|", "/")
    return f"{number:.{digits}g}" if np.isfinite(number) else str(number)


def _meanstd(values):
    finite = np.asarray([float(v) for v in values if v is not None], dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return "—"
    return f"{np.mean(finite):.6g} ± {np.std(finite, ddof=1) if len(finite) > 1 else 0:.3g}"


def _table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in rows),
    ])


def save_rollout(directory, target, audio, controls, actions, rate):
    """Save unscaled audio, actual controls and three complementary figures.

    IEEE float WAV files deliberately preserve values outside [-1, 1].  Playback
    software may clip those values, so float64 ``rollout.npz`` is the numerical
    source of truth and no listening gain is applied to the evaluation export.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = np.asarray(target, dtype=np.float64)
    audio = np.asarray(audio, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    rate = int(rate)
    if rate <= 0 or target.ndim != 1 or audio.ndim != 1 or not len(target) or not len(audio):
        raise ValueError("Expected nonempty mono audio and a positive rate")
    if not np.isfinite(target).all() or not np.isfinite(audio).all():
        raise ValueError("Cannot export nonfinite waveforms")
    if controls.ndim != 2 or controls.shape[1] != 5 or not len(controls) or not np.isfinite(controls).all():
        raise ValueError("Expected a finite N-by-5 actual control trajectory")
    np.savez_compressed(directory / "rollout.npz", target=target, audio=audio,
                        controls=controls, actions=actions, rate=np.asarray(rate))
    wavfile.write(directory / "target_raw.wav", rate, target.astype(np.float32))
    wavfile.write(directory / "generated_raw.wav", rate, audio.astype(np.float32))
    (directory / "audio_format.json").write_text(json.dumps({
        "npz_dtype": "float64", "wav_format": "IEEE float32",
        "sample_rate": rate, "independent_gain_normalization": False,
        "target_peak": float(np.max(np.abs(target))),
        "generated_peak": float(np.max(np.abs(audio))),
        "units": "model output; no calibrated sound-pressure conversion",
        "sap_evaluated": False,
    }, indent=2) + "\n")
    comparison(target, audio, rate, directory / "spectrogram.png")
    duration = len(target) / rate
    # Every physical block is 10 ms, including the substeps of constant Task A.
    # An early numerical failure must not stretch its partial history to 0.35 s.
    control_trajectories(controls, directory / "controls.png", control_s=.01)
    width = max(1, min(len(target), len(audio), round(.005 * rate)))
    kernel = np.ones(width) / width
    fig, ax = plt.subplots(figsize=(10, 4))
    try:
        for values, label in ((target, "Target"), (audio, "Generated")):
            envelope = np.sqrt(np.maximum(np.convolve(values * values, kernel, mode="same"), 0))
            ax.plot(np.arange(len(values)) / rate, envelope, label=label, linewidth=1.4)
        ax.set(xlabel="Time [s]", ylabel="5 ms RMS envelope [model output units]", xlim=(0, duration))
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(directory / "amplitude_envelope.png", dpi=150)
    finally:
        plt.close(fig)


def _landscape_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return [{key: float(value) for key, value in row.items() if value not in (None, "")}
                for row in csv.DictReader(stream)]


def _true_control(output):
    path = Path(output) / "diagnostics" / "target.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as target:
            return np.asarray(target["controls"], dtype=float)[0]
    return np.asarray([4.1e7, 3.5e7, 3.4e6, 4.9e4, 4.9e4])


def plot_landscapes(output):
    """Plot measured coarse-grid values, marking the known reachable target."""
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    true = _true_control(output)
    rows1 = _landscape_rows(output / "diagnostics" / "landscape_1d.csv")
    rows2 = _landscape_rows(output / "diagnostics" / "landscape_2d.csv")
    if rows1:
        rows1.sort(key=lambda r: r["epsilon_l"])
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        try:
            for ax, metric, title in zip(axes, METRIC_NAMES[:2], ("Full-episode acoustic error", "Balanced audit loss")):
                values = np.asarray([row[metric] for row in rows1])
                x = np.asarray([row["epsilon_l"] for row in rows1]) / 1e7
                ax.plot(x, values, ".-", markersize=5, label="Measured candidates")
                ax.axvline(true[0] / 1e7, color="crimson", linestyle="--", label="Known target control")
                best = int(np.nanargmin(values))
                ax.plot(x[best], values[best], "*", color="black", markersize=14, label="Measured minimum")
                ax.set(ylabel=title, ylim=(0, None))
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            axes[-1].set_xlabel(r"$\varepsilon_l$ [$10^7$ model units]")
            fig.tight_layout()
            fig.savefig(figures / "landscape_1d.png", dpi=160)
        finally:
            plt.close(fig)
    if rows2:
        # Plot candidates themselves: no unmeasured interpolated minimum is implied.
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
        try:
            x = np.asarray([r["epsilon_l"] for r in rows2]) / 1e7
            y = np.asarray([r["ps"] for r in rows2]) / 1e6
            for ax, metric, title in zip(axes, METRIC_NAMES[:2], ("Full-episode acoustic error", "Balanced audit loss")):
                values = np.asarray([r[metric] for r in rows2])
                finite = np.isfinite(values)
                if not finite.any():
                    ax.text(.5, .5, "No finite candidates", transform=ax.transAxes, ha="center")
                    continue
                mesh = ax.scatter(x[finite], y[finite], c=values[finite], marker="s",
                                  s=max(22, min(450, 60000 / len(rows2))), cmap="viridis", rasterized=True)
                best = int(np.nanargmin(values))
                ax.plot(x[best], y[best], "o", color="none", markeredgecolor="white", markeredgewidth=2,
                        markersize=13, label="Measured minimum")
                ax.plot(true[0] / 1e7, true[2] / 1e6, "*", color="crimson", markeredgecolor="white",
                        markersize=15, label="Known target")
                low = values <= np.quantile(values[finite], .1)
                ax.scatter(x[low], y[low], s=max(30, min(480, 65000 / len(rows2))),
                           facecolors="none", edgecolors="orange", linewidths=.8,
                           label="Lowest 10% measured")
                ax.set(title=title, xlabel=r"$\varepsilon_l$ [$10^7$ model units]")
                ax.legend(fontsize=7, loc="best")
                fig.colorbar(mesh, ax=ax, label="Error")
            axes[0].set_ylabel(r"$p_s$ [$10^6$ model units]")
            fig.tight_layout()
            fig.savefig(figures / "landscape_2d.png", dpi=160)
        finally:
            plt.close(fig)


def _load_runs(output):
    runs = []
    for path in sorted((Path(output) / "runs").glob("*/*/seed_*/summary.json")):
        summary = _json(path)
        summary["_directory"] = path.parent
        summary["_curve"] = _json(path.parent / "curve.json", [])
        runs.append(summary)
    return sorted(runs, key=lambda r: (r["task"], METHOD_ORDER.get(r["method"], 20), r["seed"]))


def _plot_curves(output, runs):
    tasks = sorted({r["task"] for r in runs})
    if not tasks:
        return
    colors = {"TD3": "#0072B2", "SAC": "#D55E00", "random_search": "#009E73", "DE": "#CC79A7"}
    fig, axes = plt.subplots(len(tasks), 2, figsize=(13, 4 * len(tasks)), squeeze=False)
    try:
        for row, task in enumerate(tasks):
            for column, metric in enumerate(METRIC_NAMES[:2]):
                ax = axes[row, column]
                for method in sorted({r["method"] for r in runs if r["task"] == task}, key=lambda m: METHOD_ORDER.get(m, 20)):
                    selected = [r for r in runs if r["task"] == task and r["method"] == method]
                    series = []
                    for run in selected:
                        points = [p for p in run["_curve"] if p.get(metric) is not None]
                        # A final evaluation can repeat the preceding episode count.
                        unique = {float(p["episodes"]): float(p[metric]) for p in points}
                        x = np.asarray(sorted(unique), dtype=float)
                        y = np.asarray([unique[key] for key in x], dtype=float)
                        if len(x):
                            series.append((x, y))
                            ax.plot(x, y, color=colors.get(method), linewidth=.6, alpha=.3)
                    if not series:
                        continue
                    start, end = max(x[0] for x, _ in series), min(x[-1] for x, _ in series)
                    grid = np.unique(np.concatenate([x[(x >= start) & (x <= end)] for x, _ in series]))
                    if not len(grid):
                        continue
                    values = np.asarray([np.interp(grid, x, y) for x, y in series])
                    mean = np.mean(values, axis=0)
                    std = np.std(values, axis=0, ddof=1 if len(values) > 1 else 0)
                    ax.plot(grid, mean, color=colors.get(method), linewidth=1.8, label=METHOD_LABELS.get(method, method))
                    ax.fill_between(grid, np.maximum(0, mean - std), mean + std, color=colors.get(method), alpha=.15)
                ax.set(title=f"Task {task}: {TASK_LABELS.get(task, task)}", xlabel="Training episodes / candidates",
                       ylabel="Full-episode acoustic error" if metric == "acoustic_error" else "Balanced audit loss", ylim=(0, None))
                if metric == "acoustic_error":
                    ax.axhline(.08, color="black", linestyle=":", linewidth=.8, label="Exploratory 0.08")
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(output) / "figures" / "learning_curves.png", dpi=160)
    finally:
        plt.close(fig)


def _oracle_rows(data, prefix=""):
    rows = []
    if isinstance(data, list):
        for i, entry in enumerate(data):
            rows.extend(_oracle_rows(entry, str(entry.get("name", i)) if isinstance(entry, dict) else str(i)))
    elif isinstance(data, dict):
        metrics = data.get("metrics", data)
        if "acoustic_error" in metrics:
            rows.append((prefix or data.get("name", "Oracle"), _number(metrics.get("acoustic_error")),
                         _number(metrics.get("balanced_error")), _number(metrics.get("parameter_nrmse"))))
        else:
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    rows.extend(_oracle_rows(value, f"{prefix}/{key}".strip("/")))
    return rows


def _export_per_seed(output, runs):
    fields = ["task", "method", "seed", "optimizer_dimension", "elapsed_s"]
    fields += [f"{stage}_{metric}" for stage in ("initial", "final") for metric in METRIC_NAMES]
    fields += [f"{stage}_{quantity}" for stage in ("training", "evaluation")
               for quantity in ("decision_steps", "episodes", "integration_s", "native_steps", "wall_s", "failures")]
    with (Path(output) / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {key: run.get(key) for key in fields[:5]}
            for stage in ("initial", "final"):
                row.update({f"{stage}_{metric}": run.get(f"{stage}_metrics", {}).get(metric) for metric in METRIC_NAMES})
            for stage in ("training", "evaluation"):
                row.update({f"{stage}_{quantity}": run.get(stage, {}).get(quantity)
                            for quantity in ("decision_steps", "episodes", "integration_s", "native_steps", "wall_s", "failures")})
            writer.writerow(row)


def report(output):
    """Generate Japanese experiment report and plots from saved public evidence."""
    output = Path(output)
    manifest = _json(output / "manifest.json", {})
    oracles = _json(output / "diagnostics" / "oracles.json", {})
    review = _json(output / "diagnostics_review.json", {})
    preservation = _json(output / "preservation_check.json", {})
    baselines = _json(output / "baselines" / "summary.json", [])
    runs = _load_runs(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    plot_landscapes(output)
    _plot_curves(output, runs)
    _export_per_seed(output, runs)
    lines = [
        "# Phase 1: 既知制御による基礎検証と手法比較", "",
        "鳴管モデル内で既知制御から生成した Case A・0.35秒の一定目標音を対象に、"
        "到達可能性、制御実装の整合性、誤差分布、TD3・SAC・ランダム探索・差分進化を確認した。"
        "Phase 2の実録音は使用していない。既存結果を上書きせず、このフォルダに実験証拠を保存した。", "",
        "## 結果の読み方", "",
        "主評価は既存 `acoustics.distance()` の **0.35秒全区間 acoustic_error**（log-spectral誤差 + "
        "5 ms RMS包絡誤差）。targetの共通振幅スケールを用い、生成音ごとのゲイン調整・位相調整は行わない。"
        "balanced_error は独立した診断値として併記する。両者の数値を同一尺度として比較しない。"
        "parameter_nrmse は学習対象の物理範囲で正規化した正解制御からの誤差で、音響的成功の必要条件ではない。", "",
        "今回の逐次RL報酬は γ=1 の全区間主評価へのprefix potential差分であり、"
        "過去の `baseline_context` や `balanced_aligned` の報酬とは異なる。"
        "過去の実験結果と今回の値を同一訓練条件としてまとめていない。"
        "0.08は探索的な比較線で、聴覚的同一性を検証済みの閾値ではない。", "",
        "## 実装整合性とOracle", "",
        "既知目標の制御値は次のとおり。1変数課題でも非学習の p_s を目標生成時の値に固定し、"
        "decode時に既定値へ戻る不整合を避ける。元モデル・RK4・遅延計算は変更しない。", "",
        _table(["ε_l", "ε_r", "p_s", "f_e,l", "f_e,r"], [[_number(value) for value in _true_control(output)]]), "",
    ]
    if all(key in manifest for key in ("physical_low", "physical_high", "fixed_controls")):
        lines += [_table(["制御", "許容下限", "許容上限", "非学習時の固定値"], [
            [label, _number(low), _number(high), _number(fixed)]
            for label, low, high, fixed in zip(
                ("ε_l", "ε_r", "p_s", "f_e,l", "f_e,r"), manifest["physical_low"],
                manifest["physical_high"], manifest["fixed_controls"])
        ]), ""]
    oracle_table = _oracle_rows(oracles)
    if oracle_table:
        lines += [_table(["再生条件", "全区間音響誤差", "balanced", "parameter NRMSE"], oracle_table), ""]
    lines += ["Oracleの再生条件・初期状態・判定値は [oracles.json](diagnostics/oracles.json)、"
              "全実験条件は [manifest.json](manifest.json) に保存した。"
              "初期状態と遅延履歴はすべて0で、リセット時に音声デシメータの状態も0とする。"
              "制御のfloat32 action変換に由来する丸めがある場合、完全一致の物理制御再生と区別する。", "",
              "## 学習前の誤差マップ", "",
              "粗い1次元・2次元の候補をそれぞれ0.35秒数値積分し、同じ全区間評価を行った。"
              "図の既知正解位置と測定最小点を比較する。2次元図の色は実際に測定した候補の値であり、"
              "橙色の枠は測定値の下位10%を表す。候補間に未測定の極小がないことは保証しない。", ""]
    map_summary = []
    for name in ("1d", "2d"):
        candidates = _landscape_rows(output / "diagnostics" / f"landscape_{name}.csv")
        if not candidates:
            continue
        best = min(candidates, key=lambda candidate: candidate["acoustic_error"])
        best_balanced = min(candidates, key=lambda candidate: candidate["balanced_error"])
        map_summary.append([name, len(candidates), _number(len(candidates) * .35),
                            _number(sum(c.get("wall_s", 0) for c in candidates)),
                            _number(best["acoustic_error"]), _number(best["epsilon_l"]), _number(best["ps"]),
                            _number(best_balanced["balanced_error"])])
    if map_summary:
        lines += [_table(["探索", "候補数", "積分時間[s]", "候補評価実時間[s]", "最小音響誤差", "その ε_l", "その p_s", "最小balanced"], map_summary), ""]
    for name in ("1d", "2d"):
        if (output / "figures" / f"landscape_{name}.png").exists():
            lines += [f"![{name} error landscape](figures/landscape_{name}.png)", ""]
    if review:
        lines += [f"診断ゲートの判定: **{'PASS' if review.get('passed') else 'FAIL'}**。"
                  f"準備段階の実時間 {_number(review.get('prepare_wall_s'))} 秒、"
                  f"Oracleの積分時間 {_number(review.get('oracle_integration_s'))} 秒。"
                  "既知正解の到達性確認のための粗い探索で、適応的な細分化や誤差マップを使う最適化器初期化は行わない。"
                  "診断が通過してから、事前に定めた予算でA→B→Cの順に進む。"
                  "詳細は [diagnostics_review.json](diagnostics_review.json)。", ""]
    lines += ["探索に含めた既知正解点が最小でも、他の低誤差領域への収束の容易さやパラメータの一意性を"
              "証明するものではない。追加細分化の有無と診断後の学習実施判断はmanifestの診断記録を参照。", "",
              "## 手法比較", "",
              "Aは1変数の一定制御（1エピソード1行動）、Bは1変数の10 msごとの制御、Cは2変数の10 msごとの制御。"
              "Aはバンディット型の別課題で、B/Cの逐次制御と同一のRL問題ではない。"
              "各課題内で同じ目標・物理範囲・全区間評価・候補音の長さを使用する。"
              "ランダム探索と差分進化はそれぞれの課題と同じ制御次元を最適化する。"
              "直接最適化の初期値は独立したランダム候補、RLの初期値は未学習の決定論的方策であり、"
              "同一の初期音を意味しない。", ""]
    settings = manifest.get("settings", {})
    if settings.get("tasks"):
        lines += [_table(["課題", "訓練episode / 候補予算", "評価間隔[episode]", "最適化次元"], [
            [task, spec.get("episodes", "—"), spec.get("eval_every_episodes", "—"),
             {"A": 1, "B": 35, "C": 70}.get(task, "—")]
            for task, spec in sorted(settings["tasks"].items())
        ]), "",
            f"seed={settings.get('seeds', [])}。RLのwarm-upは {_number(settings.get('warmup_episodes'))} "
            f"episode、SACのentropy係数設定は `{settings.get('sac_ent_coef', 'manifest参照')}`。"
            f"差分進化の個体数は {_number(settings.get('de_population'))}。"
            "B/Cの35/70次元に対してこの個体数・候補予算は小さく、差分進化の十分な収束を保証しない。"
            "各候補が独立に全区間の時間列を直接探索する方法と、観測に応じて行動する方策を学習するRLには"
            "表現と計算の違いが残る。エピソード予算を揃えても勾配更新費用や実時間は等しくない。", ""]
    if baselines:
        baseline_groups = defaultdict(list)
        for entry in baselines:
            baseline_groups[(entry["task"], entry["name"])].append(entry)
        baseline_rows = [[task, name, len(entries),
                          _meanstd([entry.get("acoustic_error") for entry in entries]),
                          _meanstd([entry.get("balanced_error") for entry in entries]),
                          _meanstd([entry.get("parameter_nrmse") for entry in entries])]
                         for (task, name), entries in sorted(baseline_groups.items())]
        lines += ["同じ評価で確認した基準音の結果を示す。neutralは正規化action=0、random_controlは"
                  "探索で選択しない単一のランダム制御で、最良候補を選択するrandom_searchとは異なる。"
                  "neutralを複数seedで再生した標準偏差0は独立した性能実験のばらつきを意味しない。", "",
                  _table(["課題", "基準", "本数", "音響誤差", "balanced", "parameter NRMSE"], baseline_rows), "",
                  "基準音の全値は [baselines/summary.json](baselines/summary.json)。", ""]
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["task"], run["method"])].append(run)
    if runs:
        rows = []
        for (task, method), selected in grouped.items():
            rows.append([task, METHOD_LABELS.get(method, method), ",".join(str(r["seed"]) for r in selected),
                         _meanstd([r["initial_metrics"].get("acoustic_error") for r in selected]),
                         _meanstd([r["final_metrics"].get("acoustic_error") for r in selected]),
                         _meanstd([r["final_metrics"].get("balanced_error") for r in selected]),
                         _meanstd([r["final_metrics"].get("parameter_nrmse") for r in selected]),
                         sum(float(r["final_metrics"]["acoustic_error"]) <= .08 for r in selected)])
        lines += ["平均 ± 標本標準偏差（seed間）。各seedの初期・最終値と計算量は [per_seed.csv](per_seed.csv)。", "",
                  _table(["課題", "手法", "seed", "初期音響誤差", "最終音響誤差", "最終balanced", "最終parameter NRMSE", "≤0.08のseed数"], rows), ""]
        for task in sorted({r["task"] for r in runs}):
            selected = [r for r in runs if r["task"] == task]
            best = min(selected, key=lambda r: r["final_metrics"]["acoustic_error"])
            below = sum(r["final_metrics"]["acoustic_error"] <= .08 for r in selected)
            lines += [f"課題{task}の最小最終音響誤差は {_number(best['final_metrics']['acoustic_error'])} "
                      f"（{best['method']}、seed {best['seed']}）。全{len(selected)} run中{below} runが探索線0.08以下だった。"
                      "最小の単一seedだけで手法の優位性を判断せず、上表の平均とばらつきを併せて読む。", ""]
        lines += ["![learning curves](figures/learning_curves.png)", "",
                  "学習曲線の横軸は訓練エピソード数（直接最適化では候補数）。太線はseed平均、帯は標本標準偏差、"
                  "細線は各seed。評価間隔が異なる場合は表示用に線形補間する。"
                  "曲線を含む評価のシミュレーション費用は訓練とは別に計上する。", ""]
    else:
        lines += ["学習runはまだ保存されていない。このレポートは実装・誤差マップの診断段階である。", ""]
    lines += ["## 計算量と資源制約", "",
              "decision_stepsはエージェント/探索器が選んだ制御行動数、episodesは生成した候補音の本数。"
              "integration_sは数値積分したモデル時間、native_stepsは元のRK4積分刻み数を表す。"
              "Aの1行動も0.35秒を積分するため、AとB/Cのstep数だけを比較して計算量が等しいとは扱わない。"
              "実時間にはPython・損失計算・方策更新などの費用も含まれ、同時処理の有無によって変動する。", ""]
    if runs:
        resource_rows = []
        for run in runs:
            train, evaluate = run.get("training", {}), run.get("evaluation", {})
            resource_rows.append([
                run["task"], run["method"], run["seed"], run.get("optimizer_dimension", "—"),
                _number(train.get("decision_steps")), _number(train.get("episodes")), _number(train.get("integration_s")),
                _number(evaluate.get("episodes")), _number(evaluate.get("integration_s")),
                _number(train.get("wall_s")), _number(evaluate.get("wall_s")), _number(run.get("elapsed_s")),
                _number(train.get("failures", 0) + evaluate.get("failures", 0)),
            ])
        lines += [_table(["課題", "手法", "seed", "次元", "訓練行動", "訓練episode", "訓練積分[s]", "評価episode", "評価積分[s]", "訓練実時間[s]", "評価実時間[s]", "run実時間[s]", "failure数"], resource_rows), ""]
        totals = []
        for stage in ("training", "evaluation"):
            totals.append([stage] + [_number(sum(float(r.get(stage, {}).get(key, 0)) for r in runs))
                                     for key in ("decision_steps", "episodes", "integration_s", "native_steps", "wall_s", "failures")])
        lines += [_table(["区分", "行動数", "episode", "積分[s]", "native刻み数", "実時間の合計[s]", "failure"], totals), "",
                  "上記合計はrun内の訓練・評価の集計。目標生成、Oracle、誤差マップ、図の描画は別費用で、"
                  "全体の実行実時間・診断費用はmanifestの記録を参照。"
                  "有限の予算で得た結果であり、未収束の結果からRL一般の不可能性は結論しない。", ""]
    lines += ["## 保存物・SAP評価への準備", "",
              "各runの `rollout.npz` はtarget/audioをfloat64で保持し、`target_raw.wav` と `generated_raw.wav` は"
              "IEEE float32で元振幅を保持する。音声の個別ピーク正規化やクリップは行わない。"
              "モデル出力単位であり、物理的な音圧への校正は行っていない。±1を超えるWAVは再生側でクリップ"
              "される場合があるため、定量解析の正本はNPZとする。", "",
              "`spectrogram.png` は既存描画関数を使用し、10 MHzへの表示用再サンプリング、"
              "nperseg=245760、noverlap=184320、Blackman–Harris、gray_r、−50〜−20 dB、0〜10 kHz。"
              "完全に含まれる窓の中心時刻のみを表示して両端の外挿を避ける。スペクトログラムのピーク正規化は"
              "表示専用なので、振幅差は共通の元単位を使う `amplitude_envelope.png` で確認する。"
              "`controls.png` は適用した5制御の時間推移を示す。", "",
              "SAPは本実験では未実施で、独立した将来の評価とする。音響誤差・balanced loss・独自の記述特徴を"
              "公式SAPスコアとは呼ばない。", "",
              "## 確認できる範囲と未解決事項", "",
              "Oracleと誤差マップは、既知制御の目標が同じモデル・初期状態・固定制御の下で再現可能かを検証する。"
              "学習結果は、その限られた到達可能な目標と訓練予算に対する制御獲得性能を示す。"
              "時変制御の方策が良好でも目標自体は一定制御で生成されており、複雑な時変目標への十分性は未検証である。"
              "パラメータ誤差が残って音響誤差が小さい場合は識別の非一意性を検討し、音響的成功を直ちに否定しない。", "",
              "この結果から実録音への一般化や左右2音源の優位性は主張しない。"
              "seed数は小さく、ハイパーパラメータ選択と計算予算の影響が残る。"
              "失敗例・未到達の閾値・数値失敗は表中に残し、成功したseedだけを選んだ結論にはしない。", "",
              "## 実験条件の完全記録", "",
              "主な追加は [検証環境](../../phase1_validation_env.py)、"
              "[実験runner](../../phase1_validation.py)、[描画・レポート](../../phase1_validation_report.py)、"
              "[実験設定](../../configs/phase1_validation.json)。"
              "固定制御の明示、全区間主評価に一致する報酬、課題別の行動次元と計算量記録を追加した。"
              "既存の元モデルと既存結果の保存先を変更しない。"
              "再実行コマンド・予算・初期状態・reward定義・診断から学習への判断は以下のmanifestで確認できる。", "",
              f"既存ファイル保存の検査: {'PASS' if preservation.get('passed') else '未完了またはFAIL'}、"
              f"対象 {_number(preservation.get('files_checked'))} ファイル。"
              "詳細は [preservation_check.json](preservation_check.json)。", "",
              "```json", json.dumps(manifest, ensure_ascii=False, indent=2), "```", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return output / "REPORT.md"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    print(report(options.output))
