"""Private-only Phase 2 report and paired figures; no public result export."""
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from .acoustics import distance
from .experiments import write_json
from .issue11 import CORE_FILES, REWARDS, json_read, run_directory
from .model import ROOT
from .objectives import BalancedConfig, BalancedObjective
from .phase2 import model_configuration, private_output
from .plotting import draw_spectrogram, plt


def fmt(values, digits=4):
    return f"{np.mean(values):.{digits}f} ± {np.std(values):.{digits}f}"


def number(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def activity_at_ratio(frames, ratio=.1):
    """Supplementary low-energy diagnosis; it does not change the reward mask."""
    et = np.asarray(frames["target_envelope"])
    eg = np.asarray(frames["generated_envelope"])
    w = np.asarray(frames["weights"])
    threshold = ratio*float(et.max())
    active = (et >= threshold) & (et.max() > 1e-8)
    generated = (eg >= threshold) & (et.max() > 1e-8)
    low = ~active
    return dict(ratio=ratio, low_energy_duration_s=float(w[low].sum()*.35),
                overproduction_low_fraction=(float(w[low & generated].sum()/w[low].sum()) if low.any() else None),
                missing_active_fraction=(float(w[active & ~generated].sum()/w[active].sum()) if active.any() else None),
                disagreement=float(w[active != generated].sum()))


def collect(output):
    protocol = json_read(output/"protocol.json")
    current_hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in CORE_FILES}
    if current_hashes != protocol["core_sha256"]:
        raise ValueError("Core source no longer matches this recording experiment")
    config = protocol["experiment"]
    rows, csv_rows, residuals = [], [], []
    expected_count = 0
    for algorithm in config["algorithms"]:
        decision = json_read(output/f"{algorithm}_extension_decision.json")
        for budget in (config["steps"], config["extension_steps"]):
            cases = config["conditions"] if budget == config["steps"] else [r["condition"] for r in decision["cases"] if r["eligible"]]
            expected_count += len(cases)*len(config["profiles"])*len(config["seeds"])
            for case in cases:
                for profile in config["profiles"]:
                    for seed in config["seeds"]:
                        path = run_directory(output, algorithm, budget, case, profile, seed)
                        cfg, result = [json_read(path/f) for f in ("config.json", "summary.json")]
                        curve, diagnosis = [json_read(path/f) for f in ("curve.json", "diagnostics.json")]
                        expected_model = model_configuration(dict(config, **REWARDS[profile]), case, gain=protocol["output_gain"])
                        assert cfg["model"] == json.loads(json.dumps(asdict(expected_model)))
                        assert cfg["core_sha256"] == protocol["core_sha256"]
                        assert cfg["algorithm"] == algorithm and cfg["seed"] == seed and cfg["profile"] == profile
                        assert cfg["target_sha256"] == protocol["target_sha256"]
                        assert result["training_steps"] == budget == cfg["settings"]["steps"]
                        assert [e["steps"] for e in curve] == list(range(0, budget+1, config["eval_interval"]))
                        assert result["evaluation_steps"] == sum(e["env_steps"] for e in curve)+result["env_steps"]
                        checkpoint_failures = sum(bool(e["failed"]) for e in curve)
                        low_energy = dict(low_energy_duration_s=None, overproduction_low_fraction=None,
                                          missing_active_fraction=None, disagreement=None)
                        with np.load(path/"rollout.npz", allow_pickle=False) as data:
                            target, audio, controls = data["target"], data["audio"], data["physical_controls"]
                            assert hashlib.sha256(target.tobytes()).hexdigest() == protocol["target_sha256"]
                            assert controls.shape[1] == 5
                            assert cfg["model"]["output_gain"] == protocol["output_gain"]
                            if case == "B":
                                np.testing.assert_array_equal(controls[:, 4], np.full(len(controls), 1e7))
                            if not result["failed"]:
                                assert len(audio) == len(target) == 17500
                                if checkpoint_failures == 0:
                                    assert result["evaluation_steps"] == (budget//config["eval_interval"]+2)*35
                                old = distance(target, audio, 50000, scale=protocol["shared_loss_scale"])
                                obj = BalancedObjective(target, 50000, scale=protocol["shared_loss_scale"])
                                balanced = obj.score(audio)
                                delta_old = abs(old["total"]-result["error"])
                                delta_new = abs(balanced["total"]-result["objective_error"])
                                assert max(delta_old, delta_new) < 1e-11
                                aligned_delta = (abs(result["discounted_reward_cost"]-result["objective_error"])
                                                 if profile == "balanced_aligned" else None)
                                if aligned_delta is not None:
                                    assert aligned_delta < 1e-11
                                residuals.append(dict(path=str(path.relative_to(output)), old=delta_old,
                                                      balanced=delta_new, aligned=aligned_delta))
                                frames = json_read(path/"activity_frames.json")
                                low_energy = activity_at_ratio(frames)
                                assert len(frames["weights"]) == len(obj.ends)
                            action_change = float(np.mean(np.diff(data["actions"], axis=0)**2)) if len(data["actions"]) > 1 else 0.0
                            control_stats = {f"{name}_{stat}": float(function(controls[:, i]))
                                             for i, name in enumerate(("epsilon_l", "epsilon_r", "p_s", "f_e_l", "f_e_r"))
                                             for stat, function in (("mean", np.mean), ("std", np.std))}
                        row = dict(algorithm=algorithm, budget=budget, condition=case, profile=profile, seed=seed,
                                   relative_path=str(path.relative_to(output)), free_controls=len(expected_model.control_indices),
                                   result=result, diagnosis=diagnosis, low_energy=low_energy,
                                   checkpoint_failures=checkpoint_failures)
                        rows.append(row)
                        motion = diagnosis.get("motion", {})
                        activity = diagnosis.get("target_activity", {})
                        csv_rows.append(dict(algorithm=algorithm, budget=budget, condition=case, profile=profile, seed=seed,
                                             free_controls=row["free_controls"], initial_legacy=result["initial_error"],
                                             final_legacy=result["error"], initial_balanced=curve[0]["objective_error"],
                                             final_balanced=result["objective_error"],
                                             legacy_log=result["metrics"]["log_spectral"], legacy_envelope=result["metrics"]["envelope"],
                                             mean_reward_cost=result["mean_reward_cost"], discounted_reward_cost=result["discounted_reward_cost"],
                                             sustained=diagnosis["sustained_acoustic"], active20ms=diagnosis["active_20ms_fraction"],
                                             band_db=diagnosis["late_band_relative_db"],
                                             left_velocity_std=motion.get("std", [None]*4)[1],
                                             right_velocity_std=motion.get("std", [None]*4)[3],
                                             missing_active=activity.get("missing_active_fraction"),
                                             false_quiet=activity.get("false_voice_quiet_fraction"),
                                             target_quiet_duration_s=activity.get("quiet_duration_s"),
                                             underproduction=activity.get("active_underproduction"),
                                             low_energy_duration_s=low_energy["low_energy_duration_s"],
                                             low_energy_overproduction=low_energy["overproduction_low_fraction"],
                                             low_energy_missing=low_energy["missing_active_fraction"],
                                             action_change_mse=action_change, training_failures=result["training_numerical_failures"],
                                             checkpoint_failures=checkpoint_failures,
                                             final_failed=result["failed"], evaluation_steps=result["evaluation_steps"], **control_stats))
    assert len(rows) == expected_count
    # Match the two profiles within algorithm/Case/seed/budget; initial policies
    # are seeded identically and must have identical common evaluation metrics.
    pairs = []
    for row in rows:
        if row["profile"] != "baseline":
            continue
        other = next(r for r in rows if all(r[k] == row[k] for k in ("algorithm", "budget", "condition", "seed"))
                     and r["profile"] == "balanced_aligned")
        assert row["result"]["initial_error"] == other["result"]["initial_error"]
        pairs.append(dict(algorithm=row["algorithm"], budget=row["budget"], condition=row["condition"], seed=row["seed"],
                          legacy_delta=other["result"]["error"]-row["result"]["error"],
                          balanced_delta=other["result"]["objective_error"]-row["result"]["objective_error"],
                          sustained_delta=int(other["diagnosis"]["sustained_acoustic"])-int(row["diagnosis"]["sustained_acoustic"])))
    unchanged = {}
    for name in ("model.py", "native/model.cpp", "native/reference.cpp", "model_env.py", "acoustics.py", "objectives.py", "plotting.py"):
        original = subprocess.check_output(["git", "show", f"{protocol['base_commit']}:Deeplearning/{name}"], cwd=ROOT)
        unchanged[name] = original == (ROOT/name).read_bytes()
    assert all(unchanged.values())
    write_csv(output/"per_seed.csv", csv_rows)
    write_csv(output/"paired_differences.csv", pairs)
    write_json(output/"analysis.json", rows)
    write_json(output/"verification.json", dict(runs=len(rows), training_steps=sum(r["result"]["training_steps"] for r in rows),
                                               evaluation_steps=sum(r["result"]["evaluation_steps"] for r in rows),
                                               paired_initial_evaluations_identical=True,
                                               target_gain_controls_budgets_and_evaluation_counts_matched=True,
                                               case_b_right_applied_force_verified=True, unchanged_from_base=unchanged,
                                               recalculation_residuals=residuals,
                                               final_runner_sha256=hashlib.sha256((ROOT/"issue11.py").read_bytes()).hexdigest(),
                                               analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return rows, pairs


def group_rows(rows):
    groups = []
    for algorithm in ("TD3", "SAC"):
        for budget in (1050, 4200):
            for case in "ABC":
                for profile in REWARDS:
                    group = [r for r in rows if (r["algorithm"], r["budget"], r["condition"], r["profile"]) == (algorithm, budget, case, profile)]
                    if not group:
                        continue
                    results, diagnoses = [r["result"] for r in group], [r["diagnosis"] for r in group]
                    groups.append(dict(algorithm=algorithm, budget=budget, condition=case, profile=profile, seeds=len(group),
                                       initial_legacy_mean=float(np.mean([r["initial_error"] for r in results])),
                                       legacy_mean=float(np.mean([r["error"] for r in results])), legacy_std=float(np.std([r["error"] for r in results])),
                                       balanced_mean=float(np.mean([r["objective_error"] for r in results])), balanced_std=float(np.std([r["objective_error"] for r in results])),
                                       sustained=sum(d["sustained_acoustic"] for d in diagnoses),
                                       active20ms_mean=float(np.mean([d["active_20ms_fraction"] for d in diagnoses])),
                                       failures=sum(r["training_numerical_failures"] for r in results),
                                       checkpoint_failures=sum(r["checkpoint_failures"] for r in group),
                                       final_failed=sum(r["failed"] for r in results), legacy_threshold=sum(r["error"] <= .08 for r in results)))
    return groups


def curve_figures(output, rows):
    for algorithm in ("TD3", "SAC"):
        for budget in (1050, 4200):
            cohort = [r for r in rows if r["algorithm"] == algorithm and r["budget"] == budget]
            if not cohort:
                continue
            with plt.style.context(["science", "ieee", "no-latex"]):
                fig, axes = plt.subplots(2, 3, figsize=(19, 10), squeeze=False)
                for col, case in enumerate("ABC"):
                    for profile, color in zip(REWARDS, ("#0072B2", "#D55E00")):
                        group = [r for r in cohort if r["condition"] == case and r["profile"] == profile]
                        if not group:
                            continue
                        curves = [json_read(output/r["relative_path"]/"curve.json") for r in group]
                        x = [e["steps"] for e in curves[0]]
                        for i, metric in enumerate(("error", "objective_error")):
                            matrix = np.array([[e[metric] for e in curve] for curve in curves])
                            for seed, values in enumerate(matrix):
                                axes[i, col].plot(x, values, color=color, alpha=.3, linestyle=("-", "--", ":")[seed])
                            axes[i, col].plot(x, matrix.mean(axis=0), color=color, linewidth=2, label=profile)
                            axes[i, col].set(xlabel="Training environment steps", ylabel="Legacy full error" if i == 0 else "Balanced full loss")
                            axes[i, col].legend(fontsize=10)
                    axes[0, col].set_title(f"Case {case}", fontsize=17)
                fig.suptitle(f"{algorithm}, {budget} steps per seed", fontsize=19)
                fig.tight_layout()
                fig.savefig(output/f"learning_curves_{algorithm}_{budget}.png", dpi=130)
                plt.close(fig)


def galleries(output, rows):
    for algorithm in ("TD3", "SAC"):
        for budget in (1050, 4200):
            for case in "ABC":
                group = [r for r in rows if r["algorithm"] == algorithm and r["budget"] == budget and r["condition"] == case]
                if not group:
                    continue
                destination = output/f"{algorithm}_{budget}/gallery_{case}.png"
                # Each panel uses the unchanged display helper, scales and crop.
                with plt.style.context(["science", "ieee", "no-latex"]):
                    fig, axes = plt.subplots(3, 3, figsize=(30, 24), sharex=True, sharey=True)
                    for seed in range(3):
                        waves = []
                        for profile in REWARDS:
                            path = run_directory(output, algorithm, budget, case, profile, seed)
                            with np.load(path/"rollout.npz", allow_pickle=False) as data:
                                target = data["target"].copy()
                                waves.append(data["audio"].copy())
                        for col, (wave, label) in enumerate(zip((target, *waves), ("Target", *REWARDS))):
                            ax = axes[seed, col]
                            draw_spectrogram(ax, wave, 50000)
                            ax.set_title(f"{label}, seed {seed}", fontsize=28)
                            if seed == 2:
                                ax.set_xlabel("Time [s]", fontsize=46)
                    fig.suptitle(f"{algorithm}, Case {case}, {budget} training steps", fontsize=32)
                    fig.tight_layout(rect=(0, 0, 1, .98))
                    fig.savefig(destination, dpi=90, bbox_inches="tight")
                    plt.close(fig)


def envelope_figure(output, rows):
    # Predetermined representative: TD3 / Case A / seed 0 at the larger budget,
    # falling back to the initial budget if that Case was ineligible.
    match = [r for r in rows if r["algorithm"] == "TD3" and r["condition"] == "A" and r["seed"] == 0]
    budget = max(r["budget"] for r in match)
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        for profile, color in zip(REWARDS, ("#0072B2", "#D55E00")):
            run = next(r for r in match if r["budget"] == budget and r["profile"] == profile)
            frames = json_read(output/run["relative_path"]/"activity_frames.json")
            time = frames["end_s"]
            if profile == "baseline":
                axes[0].plot(time, frames["target_envelope"], color="black", linewidth=2, label="Target")
                axes[1].step(time, frames["target_active"], where="post", color="gray", linestyle=":", label="Target reward mask (1%)")
                target_analysis = np.array(frames["target_envelope"]) >= .1*max(frames["target_envelope"])
                axes[1].step(time, target_analysis.astype(float), where="post", color="black", label="Target analysis mask (10%)")
            axes[0].plot(time, frames["generated_envelope"], color=color, label=profile)
            threshold = .1*max(frames["target_envelope"])
            axes[1].step(time, (np.array(frames["generated_envelope"]) >= threshold).astype(float),
                         where="post", color=color, label=f"{profile} (10% target threshold)")
        axes[0].set_ylabel("Shared-scale band envelope")
        axes[1].set(xlabel="Time [s]", ylabel="Activity / low-energy diagnosis")
        for ax in axes:
            ax.legend(fontsize=10)
        fig.suptitle(f"Predetermined example: TD3 / A / seed 0 / {budget} steps")
        fig.tight_layout()
        fig.savefig(output/"representative_envelopes.png", dpi=140)
        plt.close(fig)
    return budget


def main(output, *, render_figures=True):
    output = private_output(output)
    rows, pairs = collect(output)
    groups = group_rows(rows)
    write_csv(output/"summary.csv", groups)
    if render_figures:
        curve_figures(output, rows)
        galleries(output, rows)
        representative_budget = envelope_figure(output, rows)
    else:
        representative_budget = max(r["budget"] for r in rows if r["algorithm"] == "TD3" and r["condition"] == "A" and r["seed"] == 0)
    protocol = json_read(output/"protocol.json")
    verification = json_read(output/"verification.json")
    with np.load(output/"target_frames.npz", allow_pickle=False) as data:
        active_duration = float(np.sum(data["weights"][data["voiced"]])*.35)
        quiet_duration = .35-active_duration
    lines = ["# Issue #11 ローカル詳細レポート", "", "2026-10-30 JST。[Issue #11](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/11)。録音由来の全内容はローカル限定。GitHub・Issue・PRへ転載しない。", "",
             "## まず見るべき結果5個", "",
             "1. **このREPORT.md** — 結論、全seed集計、発声と模倣の区別、採用判断。",
             "2. [summary.csv](summary.csv) — 手法・Case・報酬・予算別の平均とseed標準偏差。",
             f"3. [TD3 Case Aの比較一覧](TD3_{representative_budget}/gallery_A.png) — 各seedで目標・baseline・balanced_alignedを同じ行に配置。代表は成績を見ずTD3/A/seed 0と指定。",
             f"4. [TD3の学習曲線](learning_curves_TD3_{representative_budget}.png) — 全Case、旧誤差と新損失、各seedと平均。",
             "5. [per_seed.csv](per_seed.csv) — 初期/最終評価、発散、発声持続、左右速度、欠落・出し過ぎ、実適用制御の全seed値。", "",
             "## 実施範囲と読み方", "",
             "許諾確認済みML14412の0.35秒のみ。TD3を第一候補、SACを参考対照として、A/B/C×baseline/balanced_aligned×seed 0/1/2を同条件比較した。鳴管は1つの器官であり、左右に2つの音源を持つ。Case間の行動自由度はA/B/C=3/4/5で、器官数の比較ではない。", "",
             f"完了{len(rows)}本、訓練合計{verification['training_steps']:,}環境step、途中と最終の評価合計{verification['evaluation_steps']:,}step。最終旧誤差≤0.08は{sum(r['result']['error'] <= .08 for r in rows)}/{len(rows)}本、操作的な持続発声は{sum(r['diagnosis']['sustained_acoustic'] for r in rows)}/{len(rows)}本。0.08は暫定値で、知覚的な模倣成功や時間周波数構造の再現の判定と同一ではない。", "",
             "表の誤差は全0.35秒・小さいほど良い。±は3 seedの母標準偏差（ddof=0）。新損失と旧誤差は単位・重みが異なり、学習報酬値を横並びにして優劣を判定しない。", "",
             "## 1. 固定した条件と追加予算", "",
             "TD3の1050step→適格Caseの4200step→SACの1050step→適格Caseの4200stepの順。各段階の仕事はCase A/B/C、baseline/balanced_aligned、seed 0/1/2の順に投入し、独立した3プロセスで実行。RNGを共有するスレッド学習は使わない。プロセス間の終了順と計算秒数は優劣の評価へ使用していない。", "",
             "全条件でpower8、同一の原区間・前処理・目標配列・出力ゲイン・行動制約・ネットワーク・評価間隔。Case Bの右外力は実適用履歴で1e7固定を全run確認した。左右の振動が抑えられたかは、値から推測せず運動統計を別に調べる。fixed/linearは今回混在させていない。", "",
             "10 ms制御・35step/episode、dt=1e−7、音声50 kHz、32×32 MLP、lr=0.0003、batch64、学習開始256、毎step更新、γ=0.99、評価は探索なし。0stepと350stepごとの評価、最終評価を別保存。TD3の学習ノイズstd0.1、SACは従来の温度係数適応を維持。", "",
             "各手法・Caseの初回6本（両報酬×全seed）に訓練中発散がなく、全チェックポイント・最終評価が正常なら、そのCaseの全6本を4200stepで最初から学習する事前規則。良いスコアのseedや報酬だけは延長しない。初回と追加のrunは別ディレクトリで、旧runと過去Issueの結果は上書きしない。", ""]
    for algorithm in ("TD3", "SAC"):
        decisions = json_read(output/f"{algorithm}_extension_decision.json")["cases"]
        lines.append(f"- {algorithm}追加対象: "+(", ".join(r["condition"] for r in decisions if r["eligible"]) or "なし")+"。詳細は同名のextension_decision.json。")
    lines += ["", f"共通ゲイン={protocol['output_gain']:.8g}、共通損失スケール={protocol['shared_loss_scale']:.8g}。既定Aのモデル音RMS÷目標RMSを一度だけ使い、Case/報酬/seedごとには合わせ直さない。PCM変換、正確な0.35秒切出し、50 kHzへのリサンプルは従来と同じ。元の振幅と音声はNPZに保持し、聴取用WAVにも全条件で同じ係数を使う（float32でピーク1を超える場合がある）。", "",
              "baselineは旧log PSD＋5 ms包絡・直近最大30 ms。balanced_alignedはIssue #10の形状・目標重み付き対数・帯域包絡・欠落・DCと既存係数をそのまま使用し、各frame寄与を一度だけ計上してγを補償する。SAP/SE/ACIを新しい報酬へ加えていない。数式は公開のIssue #10レポート、今回の実値・コードハッシュは [protocol.json](protocol.json)。", "",
              "## 2. 全seed集計", "",
              "| 手法 | step/seed | Case(自由度) | 報酬 | 初期旧誤差平均 | 最終旧誤差 | 最終新損失 | 持続発声 | 活動20 ms率 | 訓練発散 | 評価失敗(途中/最終) | 旧≤0.08 |",
              "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        lines.append(f"| {group['algorithm']} | {group['budget']} | {group['condition']}({dict(A=3,B=4,C=5)[group['condition']]}) | {group['profile']} | {group['initial_legacy_mean']:.4f} | {group['legacy_mean']:.4f} ± {group['legacy_std']:.4f} | {group['balanced_mean']:.4f} ± {group['balanced_std']:.4f} | {group['sustained']}/3 | {group['active20ms_mean']:.3f} | {group['failures']} | {group['checkpoint_failures']}/{group['final_failed']} | {group['legacy_threshold']}/3 |")
    lines += ["", "### 同じseedでの差", "",
              "差はbalanced_aligned−baselineで、旧/新とも負なら平均上の改善。paired_differences.csvには全seedの差がある。以下の『旧で良いseed』は各seedの旧誤差が小さかった数で、統計的有意差ではない。", "",
              "| 手法 | step/seed | Case | 旧誤差差（平均±std） | 新損失差（平均±std） | 旧で良いseed | 新で良いseed |", "|---|---:|---|---:|---:|---:|---:|"]
    joint_groups = []
    for algorithm in ("TD3", "SAC"):
        for budget in (1050, 4200):
            for case in "ABC":
                group = [p for p in pairs if (p["algorithm"], p["budget"], p["condition"]) == (algorithm, budget, case)]
                if not group:
                    continue
                old, new = [p["legacy_delta"] for p in group], [p["balanced_delta"] for p in group]
                lines.append(f"| {algorithm} | {budget} | {case} | {fmt(old)} | {fmt(new)} | {sum(x<0 for x in old)}/3 | {sum(x<0 for x in new)}/3 |")
                if budget == 4200 and np.mean(old) < 0 and np.mean(new) < 0:
                    joint_groups.append(f"{algorithm}/Case {case}")
    lines += ["", "## 3. 発声、欠落、低エネルギー区間、運動", "",
              "持続発声はIssue #9と同じ操作的診断。初期0.1秒を除外し、既知A基準の250–10000 Hz帯域RMS比−60 dB以上、かつ20 ms区間の80%以上が同閾値を満たすこと。絶対的な知覚上の無音や目標との一致の定義ではない。白いスペクトログラム、厳密な全ゼロ、DCオフセット、源の運動、音響出力を分ける。", "",
              f"報酬用の目標活動マスクはIssue #10の1%（振幅比−40 dB）を維持。今回の活動時間={active_duration:.6f} s、非活動時間={max(0,quiet_duration):.6f} s。マスクは目標だけから一度定義し、生成音の大きさで変わらない。雑音様音を周期性の低さだけで除外しない。", ""]
    if quiet_duration < 1e-8:
        lines += ["このマスクでは全frameが活動扱いとなる。したがって『目標の無音区間への誤発声率』はN/Aであり、誤発声がないことを示した結果ではない。背景音や弱い成分も活動となる閾値の限界を含む。", ""]
    low_duration = next((r["low_energy"]["low_energy_duration_s"] for r in rows if r["low_energy"]["low_energy_duration_s"] is not None), 0)
    lines += [f"補助的に、目標帯域包絡が最大の10%未満を『低エネルギー区間』と定義して出し過ぎを確認した。対象時間は{low_duration:.6f}秒。これは報告用の解析で、真の無音ラベルとは呼ばず、報酬マスク・係数・学習を変更していない。この補助解析は初回報酬比較の後に追加したもので、学習の事前選択基準には使っていない。対象区間が短ければ、無音区間の制御性能を広く検証したとはいえない。", "",
              "『欠落率』は目標活動frameで生成包絡が同じ目標由来閾値を下回る重み付き時間率。『欠落量』は目標活動frameでmax(1−生成包絡/目標包絡,0)の平均。『低energy過剰率』は低エネルギー区間で生成包絡が目標最大の10%以上となった率。低い欠落率だけでは音節タイミングや周波数構造の再現を保証しない。", "",
              "| 手法 | step | Case | 報酬 | 欠落率平均 | 欠落量平均 | 低energy過剰率平均 | 左速度std | 右速度std |",
              "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for g in groups:
        group = [r for r in rows if all(r[k] == g[k] for k in ("algorithm", "budget", "condition", "profile"))]
        def average_optional(values):
            values = [v for v in values if v is not None]
            return float(np.mean(values)) if values else None
        missing = average_optional([r["diagnosis"].get("target_activity", {}).get("missing_active_fraction") for r in group])
        under = average_optional([r["diagnosis"].get("target_activity", {}).get("active_underproduction") for r in group])
        over = average_optional([r["low_energy"]["overproduction_low_fraction"] for r in group])
        left = [r["diagnosis"].get("motion", {}).get("std", [0]*4)[1] for r in group]
        right = [r["diagnosis"].get("motion", {}).get("std", [0]*4)[3] for r in group]
        lines.append(f"| {g['algorithm']} | {g['budget']} | {g['condition']} | {g['profile']} | {number(missing)} | {number(under)} | {number(over)} | {fmt(left,2)} | {fmt(right,2)} |")
    lines += ["", "左右x/yの平均・標準偏差・RMSは各runのdiagnostics.json、10 msごとの全積分点の統計はrollout.npzにある。区間統計を合成するときは平均差による変動も含む。0–0.05/0.05–0.1/0.1–0.2/0.2–0.35/0.1–0.35秒のAC RMSと帯域RMSも保存した。Bの右外力固定だけを根拠に右速度が0だと断定しない。Aの右運動を音響への寄与と取り違えない。", "",
              "![代表の包絡と活動診断](representative_envelopes.png)", "",
              "## 4. 学習曲線・スペクトログラム・制御", "",
              "図の各seedは探索なしの決定的評価。学習曲線は細線が各seed、太線が平均。スペクトログラムは各seedの目標・baseline・alignedを同じ行に配置した。図ごとのpeak基準の相対dB表示なので、黒さだけで絶対音量や誤差を判断しない。", ""]
    for algorithm in ("TD3", "SAC"):
        for budget in (1050, 4200):
            if not any(r["algorithm"] == algorithm and r["budget"] == budget for r in rows):
                continue
            lines += [f"### {algorithm} / {budget}step", "", f"![共通指標の曲線](learning_curves_{algorithm}_{budget}.png)", ""]
            for case in "ABC":
                if any(r["algorithm"] == algorithm and r["budget"] == budget and r["condition"] == case for r in rows):
                    lines.append(f"- [Case {case} 全seedの並列比較]({algorithm}_{budget}/gallery_{case}.png)、[baseline seed 0制御]({algorithm}_{budget}/{case}/baseline/seed_0/controls.png)、[aligned seed 0制御]({algorithm}_{budget}/{case}/balanced_aligned/seed_0/controls.png)。")
            lines.append("")
    notes = output/"visual_notes.md"
    lines += ["## 5. 構造の確認と採用判断", ""]
    if notes.exists():
        lines += [notes.read_text(), ""]
    else:
        lines += ["並列図は確認用に保存した。視覚印象だけを成功判定には使わない。周波数構造は旧対数PSDと新形状・目標重み付き対数成分、タイミングは包絡・欠落・低エネルギー区間の過剰を併記して解釈する。", ""]
    lines += ["4200stepで旧誤差平均と新損失平均がともに小さくなった組: "+(", ".join(joint_groups) or "なし")+"。これは同一seedでの探索的な比較であり、別の録音への汎化や統計的な優位性を示すものではない。片方の指標だけの改善を『模倣成功』とは記載しない。", ""]
    decision_notes = output/"decision_notes.md"
    if decision_notes.exists():
        lines += [decision_notes.read_text(), ""]
    else:
        lines += ["採用判断では両指標・弱音部の出し過ぎ・時間周波数構造を合わせて確認する。自動集計だけでは全条件の既定報酬を変更しない。録音ごとの判断はdecision_notes.mdへローカルに追記する。", ""]
    lines += ["単一録音・3 seedの本比較だけで、全条件へ置換する根拠やモデルの表現力の限界を確定しない。観測は時刻・前行動・現在の4状態で、遅延線全体や音声フィルタ状態・目標特徴は与えない。複数目標へ条件付きで追従する能力を示す実験ではなく、この固定区間の制御軌道を学ぶ設定である。", "",
              "0.08以下の値が得られた場合も、声道フィルタ・録音ゲイン・位相の未同定と音節構造を含めて確認する。Case間は自由度と音源制約が違い、『左右2音源が有利』という一般結論へ直接変換しない。SAP等の独立評価は後続課題とし、このIssueで報酬へ追加したりPhase 3へ進めたりしていない。", "",
              "## 6. 回帰・整合性・再実行", "",
              "公開可能な合成テスト43件が成功。旧Phase 2既定の波形とstep報酬の不変、設定切替、power8端点と中央、A/B/C自由度、Bの実適用右固定、共通スケール、alignedの割引return対応、物理式・RK4・遅延・表示設定に対する既存回帰を含む。記録は公開の `Deeplearning/reports/issue11_validation.txt`。", "",
              "全runの目標ハッシュ・ゲイン・2報酬間の初期誤差・step予算・評価回数を照合し、旧/新全体損失を保存音声から再計算した。alignedの割引returnと全体損失の対応も再確認。結果は [verification.json](verification.json)。物理モデル・acoustics・objectives・model_env・plottingは基準コミットから同一で、報酬数式・係数は変えていない。", "",
              "表示は10 MHz、Blackman–Harris 245760/184320（24.576 ms/75%）、gray_r、−50〜−20 dB、0–10000 Hz、窓全体が収まるframe中心0.012288–0.331776秒。制御はε_l, ε_r, p_s, f_e,l, f_e,rの5系列・研究表記。", "",
              "```bash", "export MPLCONFIGDIR=/tmp/birdsong-mpl", "export OPENBLAS_NUM_THREADS=1", "export OMP_NUM_THREADS=1",
              "Deeplearning/.venv/bin/python -m Deeplearning.issue11 all --local-permission-confirmed",
              "```", "",
              "設定は公開の `configs/issue11.json`。段階別はprepare/train/report。録音・キャッシュ・全出力はprivate_runs/issue11内だけに置き、再実行時に元区間・コアコード・設定が異なる既存結果を再利用しない。prepareは同設定の目標図・キャッシュを再生成し、学習済みrunは同設定なら再利用する。変更した条件は新しいprivate出力先を指定する。旧runの上書きや過去Issue結果の修復は行っていない。", ""]
    detailed_text = "\n".join(lines)
    (output/"DETAILS.md").write_text(detailed_text)
    # Keep the first document short; all budgets, timing/motion tables and
    # gallery links remain available in DETAILS.md and the CSVs.
    brief = lines[:lines.index("## 1. 固定した条件と追加予算")]
    brief += ["## 判断", ""]
    if decision_notes.exists():
        brief += [decision_notes.read_text(), ""]
    else:
        brief += ["採用判断には旧/新指標と欠落・出し過ぎ・時間周波数構造の確認が必要。詳細表だけから成功とは判定しない。", ""]
    if not any(r["result"]["error"] <= .08 for r in rows):
        brief += ["今回の全runは暫定旧誤差0.08に未到達。発声の回復・指標の部分的改善と、実録音の良好な模倣成立を区別する。", ""]
    brief += ["## 最終予算の全seed集計", "",
              "各手法・Caseの最大実施予算を示す。3 seedの平均±母標準偏差。1050stepを含む全表は [DETAILS.md](DETAILS.md)、各seedの実値はper_seed.csv。", "",
              "| 手法 | step/seed | Case | 報酬 | 旧全体誤差 | balanced全体損失 | 持続発声 | 訓練発散 |",
              "|---|---:|---|---|---:|---:|---:|---:|"]
    for g in groups:
        maximum = max(r["budget"] for r in groups if r["algorithm"] == g["algorithm"] and r["condition"] == g["condition"])
        if g["budget"] != maximum:
            continue
        brief.append(f"| {g['algorithm']} | {g['budget']} | {g['condition']} | {g['profile']} | {g['legacy_mean']:.4f} ± {g['legacy_std']:.4f} | {g['balanced_mean']:.4f} ± {g['balanced_std']:.4f} | {g['sustained']}/3 | {g['failures']} |")
    visual_link = "[全図の目視所見](visual_notes.md)" if notes.exists() else "目視所見は未記録"
    brief += ["", "## 代表例と解釈の限界", "",
              f"成績で選ばないTD3/Case A/seed 0を代表とし、下図の先頭行に両報酬を並べた。残る2行は同じCaseのseed 1/2。", "",
              f"![代表Caseの全seed比較](TD3_{representative_budget}/gallery_A.png)", "",
              f"報酬用1%マスクの非活動時間は{max(0,quiet_duration):.6f}秒。非活動区間がなければ誤発声率はN/Aで、無音再現に成功したという意味ではない。補助の10%閾値による低エネルギー対象は{low_duration:.6f}秒。低エネルギー区間は真の無音ラベルと区別する。", "",
              f"[代表の包絡](representative_envelopes.png)、{visual_link}、[同じseedでの指標差](paired_differences.csv)、[整合性検証](verification.json)を併記する。音響が持続する判定、目標に必要な強度、倍音・周波数変化とタイミングの一致は別である。白い表示は厳密な全ゼロを意味しない。", "",
              "3 seed・一つの0.35秒録音で、Case間は3/4/5自由度。右外力固定だけで右運動の消失を断定せず、比較結果を器官数や両音源の普遍的優劣へ一般化しない。SAP等の新報酬やPhase 3への拡張は行っていない。", "",
              "## 再現と確認", "",
              "合成入力の回帰テスト43件成功。旧/新の全体損失、割引return、共通目標・ゲイン・初期評価・予算・評価回数とCase Bの実適用右外力を保存成果物から検証した。", "",
              "再実行は `python -m Deeplearning.issue11 all --local-permission-confirmed`（専用venv使用）。[一般的な方法と手順](../../reports/ISSUE11_REPORT.md)、[ローカル詳細と全予算の結果](DETAILS.md)を参照。旧runは別予算・別ディレクトリに保持する。", ""]
    text = "\n".join(brief)
    (output/"REPORT.md").write_text(text)
    missing = [link for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', detailed_text+text)
               if "://" not in link and not (output/link).exists()]
    if missing:
        raise ValueError("Local report references missing artifacts")
    print("Wrote Issue 11 local report; all recording-derived artifacts remain under private_runs.", flush=True)


if __name__ == "__main__":
    main(ROOT/"private_runs/issue11")
