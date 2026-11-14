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
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 registers styles

from .plotting import (CMAP, NOVERLAP, NPERSEG, PHASE0_MAX_HZ, TARGET_FS,
                       VMAX, VMIN, WINDOW_TYPE, draw_spectrogram,
                       )


ROOT = Path(__file__).resolve().parent
ALGORITHMS = ("PPO", "TD3", "SAC")
SEEDS = (0, 1, 2)
COLORS = {"PPO": "#0072B2", "TD3": "#009E73", "SAC": "#D55E00"}
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
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
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
    settings = _read(Path(output)/"config.json")
    schedule = np.arange(0, settings["steps"]+1, settings["eval_interval"])
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
            if not np.array_equal(steps, schedule):
                raise ValueError(f"{directory}: expected the registered evaluation schedule")
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
            ax.set_xlim(0, selected[0]["training_steps"])
            ax.set_ylim(-upper * .025, upper)
            ax.set_xticks(np.linspace(0, selected[0]["training_steps"], 6, dtype=int))
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
                     "initial/rollout.npz", "agent.zip", "training_trace.npz", "best.json", "best_agent.zip"):
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


def _all_seed_plot(runs, path):
    from matplotlib.lines import Line2D
    with plt.style.context(["science", "ieee", "no-latex"]), plt.rc_context({"path.simplify": False}):
        fig, ax = plt.subplots(figsize=(16, 10))
        try:
            for run in runs:
                ax.plot(run["_steps"], run["_errors"], color=COLORS[run["algorithm"]],
                        marker=("o", "s", "^")[run["seed"]], markersize=14,
                        markerfacecolor="white", markeredgewidth=2.3, linewidth=2.8, alpha=.8)
            ax.set_xlim(0, runs[0]["training_steps"])
            ax.set_xticks(np.linspace(0, runs[0]["training_steps"], 6, dtype=int))
            ax.set_ylim(0, max(.2, max(float(r["_errors"].max()) for r in runs)*1.5))
            ax.set_xlabel("Training steps", fontsize=46)
            ax.set_ylabel("Acoustic error", fontsize=46)
            ax.tick_params(labelsize=30, width=2, length=10, direction="in", top=False, right=False)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            methods = [Line2D([], [], color=COLORS[name], linewidth=2.8, label=name) for name in ALGORITHMS]
            legend = ax.legend(handles=methods, fontsize=30, loc="upper right", ncol=3, frameon=False)
            ax.add_artist(legend)
            seeds = [Line2D([], [], color="black", linestyle="None", marker=marker, markersize=14,
                            markerfacecolor="white", label=f"seed {seed}") for seed, marker in enumerate(("o", "s", "^"))]
            ax.legend(handles=seeds, fontsize=30, loc="upper right", bbox_to_anchor=(1, .91), ncol=3, frameon=False)
            fig.tight_layout()
            fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=.1)
        finally:
            plt.close(fig)


def _long_report(output, manifest, runs, grouped, aggregates, worst):
    steps = manifest["settings"]["steps"]
    interval = manifest["settings"]["eval_interval"]
    points = steps//interval+1
    checkpoints, convergence = [], []
    for run in runs:
        for point in run["_curve"]:
            convergence.append(dict(algorithm=run["algorithm"], seed=run["seed"],
                steps=point["steps"], error=point["error"], evaluation_phase=point["evaluation_phase"],
                frequency_mae_hz=point["tracking"]["frequency_mae_hz"], amplitude_mae=point["tracking"]["amplitude_mae"]))
            if point["steps"] in (4096, 10240, 51200, steps):
                checkpoints.append(convergence[-1])
    _csv(output/"convergence.csv", convergence)
    _csv(output/"budget_checkpoints.csv", checkpoints)
    old_fixed = _read(output/"legacy_fixed_reference.json")
    historical = [dict(algorithm=r["algorithm"], seed=r["seed"], old_steps=r["training_steps"],
                       old_fixed_error=r["error"], new_final_error=next(x["error"] for x in runs if (x["algorithm"], x["seed"]) == (r["algorithm"], r["seed"]))) for r in old_fixed]
    _csv(output/"old_fixed_vs_long.csv", historical)
    audit = _optional(output/"audit.json") or {}
    execution = _read(output/"execution.json")
    tests = _read(output/"test_results.json")
    preservation = _read(output/"preservation_check.json")
    lines = [
        "# Issue #14：固定目標正弦波の長期学習", "",
        f"実行日: {manifest['client_date_jst']} JST。保存パスの20261108は旧実験名を維持したもので、今回の実行日ではない。", "",
        f"**変更内容:** 訓練予算を4,096から{steps:,}環境ステップ、定期評価を512から{interval:,}ステップへ変更。"
        "通常resetで行動空間をOS由来の乱数へ再seedする挙動も、今回の設定でのみ修正した。", "",
        "**固定条件:** PPO・TD3・SAC×seed 0/1/2、同じ0.35秒の1000→2000 Hzグライド、既存の時変振幅・局所音響報酬・観測・行動・32×32ネットワーク・学習率0.0003等は維持。"
        "旧方策を継ぎ足さず9 runをゼロから学習した。主結果は途中最良ではなく最終更新後の方策。", "",
        "**最終結果（3 seed、平均±標本SD、小さいほどよい）:** " + "、".join(
            f"{name} {_meanstd_text([r['error'] for r in grouped[name]])}" for name in ALGORITHMS) + "。", "",
        "**まず見る5図:** すべてseed 0の最終方策と対応する。", "",
        "1. [3手法の学習曲線](figures/learning_curves_seed0.png)",
        "2. [PPO最終スペクトログラム](figures/PPO_seed0_final_spectrogram.png)",
        "3. [TD3最終スペクトログラム](figures/TD3_seed0_final_spectrogram.png)",
        "4. [SAC最終スペクトログラム](figures/SAC_seed0_final_spectrogram.png)",
        "5. [TD3の周波数・振幅追従](figures/TD3_seed0_controls.png)", "",
        "## 最終性能とseed間のばらつき", "",
        _table(["手法", "初期音響誤差", "最終音響誤差", "最終周波数MAE[Hz]", "最終振幅MAE"], [
            [name, _meanstd_text([r['initial_error'] for r in grouped[name]]), _meanstd_text([r['error'] for r in grouped[name]]),
             _meanstd_text([r['tracking']['frequency_mae_hz'] for r in grouped[name]]), _meanstd_text([r['tracking']['amplitude_mae'] for r in grouped[name]])] for name in ALGORITHMS]), "",
        _table(["手法", "seed", "最終誤差", "スペクトル項", "包絡項", "途中最良", "そのstep", "初回≤0.08 step"], [
            [r['algorithm'], r['seed'], _number(r['error']), _number(r['metrics']['log_spectral']), _number(r['metrics']['envelope']),
             _number(r['best_error']), r['best_steps'], _number(r['first_threshold_step'])] for r in runs]), "",
        "[per_seed.csv](per_seed.csv) は初期/最終MAE・最大誤差・実更新回数等、[aggregate.csv](aggregate.csv) は全3 seed集計。"
        "到達stepは評価グリッドで初めて観測された時点で、全学習stepの真の初到達を特定した値ではない。"
        "0.08は探索的な参考値であり聴覚的同一性の合格基準ではない。発表図に補助線は描いていない。", "",
        "## 同一run内の予算と収束傾向", "",
        "![学習曲線](figures/learning_curves_seed0.png)", "",
        f"生の{points}点（0, {interval:,}, …, {steps:,}）を平滑化せず結んだ。終点のみ最終更新後へ揃え、"
        "更新前の同step callback値もcallback_curve.jsonに残した。初期・途中・最終前callback・最終更新後を合わせた評価回数と、表示曲線の点数は別。", "",
        _table(["手法", "4,096", "10,240", "51,200", f"{steps:,} 最終"], [
            [name]+[_meanstd_text([x['error'] for x in checkpoints if x['algorithm']==name and x['steps']==n]) for n in (4096,10240,51200,steps)] for name in ALGORITHMS]), "",
        "同じrun内の4,096→最終を比較して予算増加時の推移を読む。途中点はそのstepの更新前callback、終点は最終更新後である。"
        "全51点の詳細は[convergence.csv](convergence.csv)、予算別抜粋は[budget_checkpoints.csv](budget_checkpoints.csv)。"
        "[全seed補足図](figures/learning_curves_all_seeds.png)では色を手法、○/□/△をseedとした。", "",
    ]
    for name in ALGORITHMS:
        selected = grouped[name]
        increases = [int(np.count_nonzero(np.diff(r['_errors']) > 0)) for r in selected]
        best_final = [r['error']-r['best_error'] for r in selected]
        lines += [f"- {name}: 50区間中の誤差増加回数はseed順に{increases}。最終誤差−記録最良誤差は{[_number(v) for v in best_final]}。"
                  "この有限予算の曲線だけで数学的収束や追加予算での挙動は保証しない。"]
    lines += ["", "途中最良は初期値を含む反復in-target評価で選択した補助結果。選択バイアスを含むため主結果へ置き換えない。"
              "各runに最終agent.zipとbest_agent.zipだけを残し、全評価時点のモデルは保存していない。", "",
        "## 条件・音響評価・制御追従", "",
        f"{steps:,}はエピソード数ではなく制御行動数であり、{steps//35:,}完走エピソード＋{steps%35}部分ステップ。"
        f"評価は{points+1}回×35行動={35*(points+1):,}環境ステップ/runを学習と分離した。", "",
        "音声48 kHz、10 msごと35行動、周波数500–3000 Hz・振幅0–1。目標振幅は0.6*(0.8+0.2*cos(2πk/34))で一定化していない。"
        "resetで位相を0へ戻し、480サンプル区間間で位相連続。観測には時刻、現在の目標f/A、直前行動が含まれる。", "",
        "報酬は従来どおり10 ms局所音響誤差の負値、平滑化罰0。主評価は完全な0.35秒波形の対数スペクトル誤差＋約5 ms RMS包絡誤差。"
        "35個の局所報酬和とは同一でない。ゲイン・位相合わせ・独立音量正規化は追加していない。", "",
        *[f"- {name}: [最終スペクトログラム](figures/{name}_seed0_final_spectrogram.png)、[周波数・振幅制御](figures/{name}_seed0_controls.png)。" for name in ALGORITHMS], "",
        "図はSciencePlots、既存0–2500 Hz、10 MHzへの表示専用再サンプリング、Blackman–Harris窓、nperseg245760・noverlap184320、gray_r、−50〜−20 dB、完全な窓中心を維持。"
        "図中タイトル・seed表記は追加せず、実値の制御図とMAEを振幅一致の判断に使う。", "",
        _oracle_text(output), "",
        "## 旧4,096ステップ結果と乱数再現性", "",
        _table(["手法", "旧固定目標4,096", f"今回最終{steps:,}"], [[name,
            _meanstd_text([r['error'] for r in old_fixed if r['algorithm']==name]), _meanstd_text([r['error'] for r in grouped[name]])] for name in ALGORITHMS]), "",
        "旧値は2026-11-08の保存済み固定目標実験。旧OS由来のaction-space再seed、ライブラリ版の違いも含む参考比較であり、"
        "予算だけの因果効果として扱わない。旧ランダム目標版は書き換えず、参考CSV old_vs_fixed_target.csv に保存値を引用した。", "",
        "今回の設定は明示seedのresetで行動空間を初期化し、通常resetでは再seedしない。旧公開設定のdefaultは従来のまま。"
        "同一seedのTD3を2回320ステップ（warm-upと更新の両方を含む）走らせ、初期軌道・全行動/報酬・評価ログ・最終重みの一致をテストした。"
        "この回帰テストは本比較の9 runと別計上。全102,400ステップの反復実行は行っておらず、異なるハードウェア・版間のbitwise再現は主張しない。", "",
        "## 計算費用・更新回数・監査", "",
        _table(["手法", "seed", "訓練step", "SB3更新カウンタ", "実optimizer呼出", "訓練秒", "評価秒"], [[r['algorithm'],r['seed'],r['training_steps'],r['sb3_n_updates'],json.dumps(r['optimizer_steps']),_number(r['training_wall_s']),_number(r['evaluation_wall_s'])] for r in runs]), "",
        "PPOのSB3カウンタはepoch数、TD3/SACはgradient iteration数。実Adam呼出・環境step・評価回数とは区別した。"
        f"CPU各1スレッド、独立{manifest['workers']}プロセスで実行。並列化と監視を含むため厳密な速度順位の測定ではない。", "",
        f"9 run合計: 学習{execution['training_steps']:,}、評価{execution['evaluation_steps']:,}環境ステップ。"
        f"全run実行wall時間{execution['execution_wall_s']:.3f}秒。生成音長は学習{execution['training_generated_s']:.2f}秒・評価{execution['evaluation_generated_s']:.2f}秒で実時間とは別。", "",
        f"関連{tests['tests_passed']}テストPASS、独立監査{'PASS' if audit.get('passed') else 'FAIL'}（{audit.get('checks')}確認）、"
        f"保護照合{'PASS' if preservation.get('passed') else 'FAIL'}（{preservation.get('files_checked')}ファイル）。"
        "有限値・報酬・損失の集計は各summaryに記録。評価前後でpolicy、entropy、optimizer、利用済みbuffer内容、Python/NumPy/Torch・環境/行動空間のRNG等を署名照合した。", "",
        "最終モデルを独立に読み込み、35行動を生成して最終NPZ・WAV・音響誤差・制御・曲線終点・summaryの一致を監査した。"
        "途中で保存しない波形の指標は独立再計算できないため、記録された値と不変性検査を根拠とする。", "",
        "独立監査は一時成果と置換後に実施し、報告処理の修正後にも再点検した。学習・run内評価の費用とは分けた。"
        "開発中の2 runに対する監査検証も追加学習0、方策評価70ステップとして development_audit.json に別記した。", "",
        "## 保存先の置換・復元・再実行", "",
        "旧成果の全追跡ファイルをGitの内容と照合し、一時的な無視領域で学習・監査・報告を完了後、同じ保存パスへ入れ替えた。"
        "未追跡ファイルがあれば置換しない。失敗時は旧成果を維持/復元する。旧commit・対象ファイルhash・復元手順は[legacy_recovery.json](legacy_recovery.json)。"
        "ソース全文の重複コピーは省き、base_commit・source_hashes・source_changes.patchで実行版を記録した。", "",
        "リポジトリ直下から（指定の既存結果を、検証後に更新する）:", "", "```bash", manifest['reproducible_command'], "```", "",
        "## 研究上の限界", "",
        "これは目標f/Aを観測できる固定正弦波の追従制御である。学習と同じ音を反復評価するin-target成績であり、held-out評価や未知音への汎化ではない。"
        "3 seedのばらつきを含む有限予算の結果から手法の普遍的優劣、鳴管での有効性、実録音の模倣、左右2音源の相互作用は主張しない。"
        "報酬設計・パラメータ探索・事後の追加学習は行っておらず、Phase 1/2と鳴管C++モデルにも変更を加えていない。", "",
    ]
    (output/"REPORT.md").write_text("\n".join(lines))


def generate_report(output: Path):
    """Read nine completed fixed-target runs; write only new report artifacts."""
    started = time.perf_counter()
    output = Path(output)
    manifest = _read(output / "manifest.json")
    # Include newly added untracked source files as real patches, too. git diff
    # alone omits them until a user chooses to commit this work.
    patch = subprocess.check_output(["git", "diff", manifest["base_commit"], "--", *manifest["source_hashes"]], cwd=ROOT, text=True)
    for filename in manifest["source_hashes"]:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", filename], cwd=ROOT, capture_output=True)
        if tracked.returncode:
            delta = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", str(Path(ROOT.name)/filename)], cwd=ROOT.parent, capture_output=True, text=True)
            if delta.returncode not in (0, 1):
                raise RuntimeError(delta.stderr)
            patch += delta.stdout
    (output/"source_changes.patch").write_text(patch)
    runs = _load(output)
    sources, correspondence = _source_metadata(output, runs)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_settings = _learning_plot(runs, figures / "learning_curves_seed0.png")
    figure_records = [dict(path="figures/learning_curves_seed0.png", stage="plotted evaluations; final endpoint post-update",
                           source_curves=[f"{name}_0/curve.json" for name in ALGORITHMS], **plot_settings)]
    all_curves = {name: {run["seed"]: (run["_steps"], run["_errors"])
                        for run in runs if run["algorithm"] == name} for name in ALGORITHMS}
    _all_seed_plot(runs, figures / "learning_curves_all_seeds.png")
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
                                       stage="final", training_steps=run["training_steps"], titles=[],
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
        row.update(best_error=run.get("best_error"), best_steps=run.get("best_steps"), first_threshold_step=run.get("first_threshold_step"), optimizer_steps=json.dumps(run.get("optimizer_steps", {})))
        row.update({"initial_"+key: value for key, value in run["initial_tracking"].items()})
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
        learner_signature_scope_limits=["Recorded before/after digests cover valid buffer content, optimizer, RNG and SAC entropy parameters; not a proof of all Python object internals"],
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

    _long_report(output, manifest, runs, grouped, aggregates, worst)
    recovery = _read(output/"legacy_recovery.json")
    recovery["restore_into_empty_directory_command"] = (
        f"git archive {recovery['commit_sha']} {recovery['path']} | tar -x -C /path/to/empty/recovery-directory")
    recovery["new_only_relative_files"] = sorted(
        str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()
        and str(Path(recovery["path"])/p.relative_to(output)) not in recovery["tracked_files"])
    _write_json(output/"legacy_recovery.json", recovery)
    _write_json(output/"completion_verification.json", dict(
        passed=True, workflow=manifest["workflow"], training_steps=manifest["settings"]["steps"],
        files_sha256={str(p.relative_to(output)): _sha(p) for p in sorted(output.rglob("*"))
                      if p.is_file() and p.name != "completion_verification.json"},
        note="Exact known publication; replacement refuses any added, removed or edited file."))
    return output / "REPORT.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(generate_report(args.output))
