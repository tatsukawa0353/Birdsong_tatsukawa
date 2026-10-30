"""Build a public report using only Issue 10 synthetic experiment artifacts."""
import csv
import hashlib
import json
import subprocess

import numpy as np

from .acoustics import distance
from .experiments import write_json
from .model import ROOT
from .objectives import BalancedObjective
from .plotting import plt

OUTPUT = ROOT/"results/issue10"
PROFILES = ("baseline", "balanced_context", "balanced_aligned")


def read(path):
    return json.loads((OUTPUT/path).read_text())


def fmt(values, digits=4):
    return f"{np.mean(values):.{digits}f} ± {np.std(values):.{digits}f}"


def learning_figure(directory, rows):
    """Separate algorithms and metrics; show each seed plus the group mean."""
    algorithms = list(dict.fromkeys(r["algorithm"] for r in rows))
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, axes = plt.subplots(2, len(algorithms), figsize=(7*len(algorithms), 9), squeeze=False)
        for col, algorithm in enumerate(algorithms):
            for profile, color in zip(PROFILES, ("#0072B2", "#D55E00", "#009E73")):
                group = [r for r in rows if r["algorithm"] == algorithm and r["profile"] == profile]
                curves = [read(f"{directory}/{profile}/{algorithm}_{r['seed']}/curve.json") for r in group]
                if not curves:
                    continue
                steps = [e["steps"] for e in curves[0]]
                for row, metric in enumerate(("error", "objective_error")):
                    values = np.array([[e[metric] for e in curve] for curve in curves])
                    for seed, value in enumerate(values):
                        axes[row, col].plot(steps, value, color=color, alpha=.3,
                                            linestyle=("-", "--", ":")[seed % 3])
                    axes[row, col].plot(steps, values.mean(axis=0), color=color, label=profile, linewidth=2)
                    axes[row, col].set(xlabel="Training environment steps",
                                       ylabel="Legacy full loss" if row == 0 else "Balanced full loss")
                    axes[row, col].legend(fontsize=10)
            axes[0, col].set_title(algorithm, fontsize=16)
        fig.tight_layout()
        fig.savefig(OUTPUT/directory/"comparison_curves.png", dpi=150)
        plt.close(fig)


def verify_and_export(cohorts):
    rows, aligned_residuals, legacy_residuals, hashes = [], [], [], set()
    for directory, cohort in cohorts:
        for run in cohort:
            result, voice = run["result"], run["voice"]
            relative = f"{directory}/{run['profile']}/{run['algorithm']}_{run['seed']}"
            cfg = read(f"{relative}/config.json")
            hashes.add(cfg["target_sha256"])
            assert cfg["algorithm"] == run["algorithm"] and cfg["seed"] == run["seed"]
            assert cfg["settings"]["steps"] == run["budget"]
            assert cfg["model"]["condition"] == "A" and cfg["model"]["duration"] == .35
            assert result["training_steps"] == run["budget"]
            assert result["requested_steps"] == run["budget"]
            assert result["evaluation_steps"] == (run["budget"]//350+2)*35
            assert cfg["model"]["control_indices"] == [0, 2]
            with np.load(OUTPUT/relative/"rollout.npz", allow_pickle=False) as data:
                target, audio = data["target"], data["audio"]
                assert len(target) == len(audio) == 17500
                assert hashlib.sha256(target.tobytes()).hexdigest() == cfg["target_sha256"]
                original = distance(target, audio, 50000, scale=max(float(abs(target).max()), 1e-8))
                legacy_residuals.append(abs(original["total"]-result["error"]))
                balanced = BalancedObjective(target, 50000).score(audio)
                assert abs(balanced["total"]-result["objective_error"]) < 1e-11
                controls = data["physical_controls"]
                change = float(np.mean(np.diff(data["actions"], axis=0)**2))
                assert controls.shape == (35, 5)
            if run["profile"] == "balanced_aligned":
                aligned_residuals.append(abs(result["discounted_reward_cost"]-result["objective_error"]))
            row = dict(target=run["target"], algorithm=run["algorithm"], profile=run["profile"],
                       seed=run["seed"], budget=run["budget"], legacy_error=result["error"],
                       legacy_log=result["metrics"]["log_spectral"], legacy_envelope=result["metrics"]["envelope"],
                       balanced_error=result["objective_error"], mean_reward_cost=result["mean_reward_cost"],
                       discounted_reward_cost=result["discounted_reward_cost"],
                       sustained=voice["sustained_acoustic"], active_fraction=voice["active_20ms_fraction"],
                       late_band_db=voice["late_band_relative_db"],
                       left_velocity_std=voice["motion"]["std"][1], right_velocity_std=voice["motion"]["std"][3],
                       action_change_mse=change, training_failures=result["training_numerical_failures"],
                       evaluation_failed=result["failed"], evaluation_steps=result["evaluation_steps"],
                       first_threshold_step=result["first_threshold_step"])
            for i, name in enumerate(("epsilon_l", "epsilon_r", "p_s", "f_e_l", "f_e_r")):
                row[f"{name}_mean"] = float(controls[:, i].mean())
                row[f"{name}_std"] = float(controls[:, i].std())
            rows.append(row)
    assert max(legacy_residuals, default=0) < 1e-12
    assert max(aligned_residuals, default=0) < 1e-11
    assert len(hashes) == 1  # All current training cohorts use the same constant target.
    with (OUTPUT/"per_seed.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verification = dict(runs=len(rows), training_steps=sum(r["budget"] for r in rows),
                        evaluation_steps=sum(r["evaluation_steps"] for r in rows),
                        target_sha256=sorted(hashes),
                        max_legacy_recalculation_difference=max(legacy_residuals, default=0),
                        max_aligned_return_difference=max(aligned_residuals, default=0),
                        matched_budgets_controls_targets_and_evaluation_counts=True)
    write_json(OUTPUT/"verification.json", verification)
    return verification


def provenance():
    paths = ["issue10.py", "issue10_report.py", "objectives.py", "model_env.py", "experiments.py",
             "acoustics.py", "model.py", "native/model.cpp", "native/reference.cpp", "plotting.py", "configs/issue10.json",
             "requirements-lock.txt", "tests/test_issue10.py"]
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths if (ROOT/p).is_file()}
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    unchanged = {}
    for path in ("acoustics.py", "model.py", "plotting.py", "native/model.cpp", "native/reference.cpp"):
        file = ROOT/path
        if not file.is_file():
            continue
        old = subprocess.check_output(["git", "show", f"HEAD:Deeplearning/{path}"], cwd=ROOT)
        unchanged[path] = old == file.read_bytes()
    assert all(unchanged.values())
    write_json(OUTPUT/"provenance.json", dict(base_commit=base, code_sha256=hashes,
                                             unchanged_from_base=unchanged,
                                             data_source="ModelEnv self-generated only; no recording input",
                                             device="CPU", integration_dt=1e-7))


def main():
    config = read("config.json")
    cohorts = []
    for budget in (config["steps"], config["extension_steps"]):
        directory = f"constant_{budget}"
        if (OUTPUT/directory/"summary.json").exists():
            cohort = read(f"{directory}/summary.json")
            algorithms = (config["algorithms"] if budget == config["steps"]
                          else read("extension_decision.json")["selected_algorithms"])
            expected = {(a, p, s) for a in algorithms for p in config["profiles"] for s in config["seeds"]}
            actual = {(r["algorithm"], r["profile"], r["seed"]) for r in cohort}
            if actual != expected or len(cohort) != len(expected):
                raise ValueError(f"Incomplete or duplicated cohort: {directory}")
            cohorts.append((directory, cohort))
            learning_figure(directory, cohort)
    verification = verify_and_export(cohorts)
    provenance()
    scores = [r for r in read("audit/scores.json") if r["profile"] == "baseline"]
    activity = read("audit/activity_cases.json")
    temporal = read("audit/temporal_audit.json")
    extension = read("extension_decision.json")
    glide = read("glide_decision.json")
    baseline = read("constant_baselines/summary.json")
    frames = read("audit/target_frames.json")
    rows = [r for _, cohort in cohorts for r in cohort]
    successes = sum(r["result"]["error"] <= config["threshold"] for r in rows)
    sustained = sum(r["voice"]["sustained_acoustic"] for r in rows)
    best = min(rows, key=lambda r: r["result"]["error"])
    lines = ["# Issue #10 対応レポート", "", "2026-10-30 JST。対象: [Issue #10](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/10)。", "",
             "合成音の固定診断、差し替え可能な報酬候補、SAC/TD3の同条件比較、直接探索、回帰検証を実施した。今回のPhase 1定数目標・既定設定では、現行報酬が通常の周波数ずれ有声音より無音を優遇する仮説は再現しなかった。ただし窓を延ばす感度診断では、より遠い有声音との順位逆転が起きた。即時報酬と全体誤差の順位逆転も再現した。", "",
             f"最終の従来誤差0.08以下は全{len(rows)}学習中{successes}本、持続発声は{sustained}/{len(rows)}本。発声の成立と模倣収束を分けて扱う。新損失上の改善だけを理由に従来報酬を置き換える根拠は得ていない。", "",
             "実録音を読まず、Case Aの既知モデル制御で生成した0.35秒だけを使用した。今回の数値・図・WAV・モデルは全て合成由来。`private_runs/` の過去結果は再計算・転載していない。GitHubへの投稿・push・Issueクローズは行っていない。", "",
             "## 1. 学習前の固定診断", "",
             "全条件に目標ピークの共通スケールを使用。oracleは同一配列、wrong_voicedはε_l=3e7・p_s=3.4e6、far_voicedはε_l=5e7・p_s=4e6。正解はε_l=4.1e7・p_s=3.4e6。外力は49,000固定。matched_rmsは音量差の影響を見る固定診断用の波形だけを目標RMSへ合わせたもので、学習・通常評価の独立音量正規化ではない。DCは目標の平均値だけを出力する。遅延は先頭20 msをゼロにして末尾を切り、位相合わせ・伸縮はしない。suppressed_modelのみ診断用に左外力を5e5へ上げた。これはRLの行動自由度へは加えていない。", "",
             "| 固定波形 | 従来log PSD | 従来包絡 | 従来全体 | 従来30 ms平均cost | 従来割引cost | 新全体 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in scores:
        m = r["legacy"]
        lines.append(f"| {r['case']} | {m['log_spectral']:.5f} | {m['envelope']:.5f} | {m['total']:.5f} | {r['mean_cost']:.5f} | {r['discounted_cost']:.5f} | {r['balanced']['total']:.5f} |")
    lines += ["", "costは小さいほど良く、報酬はその負値。割引costは `-Σ γ^(t−1) r_t / 35`、γ=0.99。異なる損失の数値の大小を直接比較しない。", "",
              "無音の従来全体誤差0.64620は、wrong_voicedの0.26479、far_voicedの0.57005より大きい。対数PSDは全帯域・全フレームの等重み平均なので、目標のエネルギーが少ないビンが多いと無音のスペクトル罰は薄まり得る。しかし今回の共有ゲインでは包絡の欠落も加わり、無音有利とはならなかった。極端な10倍音量は無音より悪いが、それだけで無音の抜け道とは判定しない。Issue #9の実録音・外力可変条件への一般化はできない。", "",
              "![固定波形の報酬順序](../results/issue10/audit/reward_order.png)", "",
              "![周波数ずれ有声音](../results/issue10/audit/wrong_voiced.png)", "",
              "全固定波形は [audit/](../results/issue10/audit/) のNPZ、全項目は [scores.json](../results/issue10/audit/scores.json)、35区間別は [step_metrics.json](../results/issue10/audit/step_metrics.json)。", "",
              "### 目標の有声・無音・雑音様区間", "",
              "目標のみから帯域包絡を求め、最大値の1%（振幅比−40 dB）以上を活動フレームとする。全体最大1e−8以下なら全て非活動。生成音はマスクへ影響しない。帯域フィルタと包絡の遅れ・減衰を含む操作的マスクであり、手動の真の音節境界や周期性の検出ではない。", "",
              "| 目標・生成の組合せ | 目標活動frame率 | 従来全体 | 新全体 |", "|---|---:|---:|---:|"]
    for r in activity:
        lines.append(f"| {r['case']} | {r['voiced_frame_fraction']:.3f} | {r['baseline']['total']:.5f} | {r['balanced']['total']:.5f} |")
    lines += ["", "gatedは0.10–0.25秒だけ鳴る目標。正しい無音は誤差0、無音目標への誤発声・gatedに対する常時発声は正の誤差となる。雑音目標も活動として保持し、完全再現は0、欠落は正の誤差。常時発声ボーナスは加えていない。", "",
              "![音節タイミングの診断](../results/issue10/audit/gated_timing.png)", "",
              "### 旧報酬の感度と時間集計", "",
              "一度に1設定だけ変えた合成診断。包絡の5 ms幅は維持し、表のenvelope_weightはその係数。shared_scaleは目標・生成に同じ倍率を適用する。主設定・初期12波形・候補の重み・窓等の感度条件は学習前に固定した。far_matched_rmsは音量と周波数差の混同を調べる追加診断であり、学習後に追加したが、学習設定・損失重み・予算や移行基準の変更には使用していない。", "",
              "| 変更 | wrong voiced | far voiced | far RMS一致 | 無音 | DC | half gain | 20 ms遅延 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    sensitivity = read("audit/sensitivity.json")
    for setting in dict.fromkeys(r["setting"] for r in sensitivity if "metrics" in r):
        values = {r["case"]: r["metrics"]["total"] for r in sensitivity if r["setting"] == setting}
        lines.append("| "+setting+" | "+" | ".join(f"{values[n]:.5f}" for n in ("wrong_voiced", "far_voiced", "far_matched_rms", "silence", "dc_only", "half_gain", "delayed"))+" |")
    lines += ["", "フロアはPSDへ加算してから対数化し、その絶対dB値で割るため、単なる表示下限ではない。STFTの窓・周波数帯域・共通ゲイン・包絡重みは物理的な区別の相対重みと順位を変え得る。窓49.152 msでは無音約0.5368がfar_voiced約0.6205より良くなる。窓を長くすると帯域ビンへの成分の広がりと対数誤差が変わるためで、全設定で無音優遇が否定されたわけではない。ただし同じfar波形をRMS一致にすると約0.3152で無音より良くなり、音量差が順位逆転に寄与していたことが分かる。純粋な周波数差だけの効果とは断定しない。表示の[-50,-20] dBはこの計算へ渡していない。感度の全成分と欠落重み0/1/2/4の診断は [sensitivity.json](../results/issue10/audit/sensitivity.json)。", "",
              f"旧法は10 msごとに最大30 msを切り出し、24.576 ms窓でゼロ埋めする。最初の2区間は全frameがパディングを含み、以降は{temporal['baseline_context_frames'][2]['padded_frames']}/{temporal['baseline_context_frames'][2]['frames']}frameが含む。隣接区間では過去音声を再び評価し、切り出し端と包絡も毎回変わる。全0.35秒の旧誤差と同じ集計ではない。", "",
              f"固定群で全体と割引即時costの順位が逆転した組は `{temporal['baseline_rank_reversals']}`。wrong_voicedとhalf_gainでは全体はhalf_gainが良く、即時報酬はwrong_voicedが良い。新法は物理的な窓の重複を残すが、{temporal['balanced_unique_frames']}個のframeをそれぞれ1回だけ割り当て、重み合計を1とする。開始前の既知のゼロは左だけに補い、未生成の未来を右ゼロ埋めしない。末尾1 sample分のframeを通常frameと等重みにはせず、1/17500で重み付けする。", "",
              "## 2. 差し替え可能な候補の定義", "",
              "`acoustics.distance` と既定の報酬は保持した。`ModelConfig.loss_kind` と `reward_timing` で以下を選ぶ。baselineにも診断専用の新損失を計算するが、波形・元報酬へは影響しないことを回帰検証した。", "",
              "| profile | 特徴・損失 | step報酬 |", "|---|---|---|",
              "| baseline | 旧log PSD + 5 ms包絡 | 直近最大30 msの旧costの負値 |",
              "| balanced_context | 下記の新損失 | 同じ新frame列の直近30 ms重み付き平均の負値 |",
              "| balanced_aligned | 同じ新損失 | 新たに完成したframeの寄与を1回だけ報酬化、割引を補償 |", "",
              "baseline→balanced_contextは帯域フィルタ、因果窓、正規化、重みも同時に変わる候補全体の比較。balanced_context→balanced_alignedは特徴と重みを保ち、時間集計だけを変える比較。3条件は完全な要因計画ではなく、新損失の各成分の因果効果を単独では特定しない。", "",
              "### 数式と正規化", "",
              "目標ピークSで両波形を割り、6次Butterworth帯域通過250–10000 Hzを因果適用する。フィルタ初期状態は先頭値に対する定常状態とし、DCだけの入力から開始過渡の帯域成分を作らない。50 kHz、Blackman–Harris窓1229 sample、hop307 sample（丸め後）、左だけゼロ補完。STFTは窓和で割り、片側powerを2倍する。末尾の追加frameを含める。", "",
              "frame kの終端e_k、直前終端e_(k−1)、波形長Nについて w_k=(e_k−e_(k−1))/N。帯域power P、正規化形状q=P/Σ_f P（power総和≤1e−20ならq=0）を使う。目標のみの活動マスクをm_kとする。目標・生成の因果5 ms RMSを新たにカバーした区間で平均した値がE^T_k,E^G_k。E_ref=max(max_k E^T_k,0.05)。生波形の同区間平均をD_kとする。", "",
              "```text",
              "C_k = m_k mean_f |cumsum_f(q^T_k - q^G_k)|",
              "J_k = m_k sum_f q^T_k |10log10(P^T_k+1e-8) - 10log10(P^G_k+1e-8)| / 80",
              "A_k = |E^T_k - E^G_k| / E_ref",
              "M_k = m_k max(1 - E^G_k/max(E^T_k,1e-12), 0)^2",
              "Derr_k = |D^T_k - D^G_k|",
              "ell_k = C_k + 0.25 J_k + A_k + 2 M_k + 0.1 Derr_k",
              "L = sum_k w_k ell_k",
              "```", "",
              "Cは周波数累積分布の距離で、無出力にはゼロ分布を拡張使用する。Hz単位の周波数誤差ではない。Cの形状正規化は音量を消すが、絶対包絡A、目標有声時の欠落M、DC差が残る。大音量は包絡差が増え、無音目標への誤発声もAで罰する。極小出力と厳密な0のCには閾値の不連続があり得るが、欠落Mは残る。重みは今回の設計値で、知覚評価に合わせた校正値ではない。", "",
              "alignedではstep tで新規frameのΔL_tを計算し、r_t=−35 ΔL_t / γ^(t−1)。したがって `−Σ γ^(t−1)r_t/35=L` が恒等的に成立する（今回の平滑化罰は0）。γ=0.99を変更せず、最大の割引補償は約1.41。単純な未割引報酬平均はLとは一致しないため両方保存する。報酬尺度も変わるためSACのエントロピー係数適応等との相互作用は分離していない。", "",
              f"目標frame情報は [target_frames.json](../results/issue10/audit/target_frames.json)。共通スケールS={frames['shared_scale']:.6g}、E_ref={frames['energy_reference']:.6g}。実装は [objectives.py](../objectives.py)、[model_env.py](../model_env.py)。", "",
              "## 3. 同一予算・seedの学習比較", "",
              "Case A・ε_lと共通p_sの2行動・35step/episode・0.35秒、同一の目標配列と共通ゲイン。ε_rと左右外力は従来の既定値。SAC/TD3、seed 0/1/2、32×32 MLP、CPU1スレッド、lr=0.0003、batch64、学習開始256、毎step更新、γ=0.99。TD3探索ノイズ0.1、SACの温度係数はSB3の既定適応。評価は探索なしで0stepと350stepごと、最終評価を別に保存する。RLのstep予算に評価呼出は含めない。", "",
              "1050stepを先に実施。alignedが新しい全体損失を初期値から2/3 seed以上で改善し、同予算baselineの新損失平均も下回った手法だけを追加対象とする基準と4200step予算を、初回学習前に [issue10.json](../configs/issue10.json) へ固定した。選ばれた手法は全3profile・全seedを4200stepで最初から学習する。良いseedだけの継ぎ足しではないが、初回結果による手法選択を含む探索的実験である。", "",
              f"追加対象: {', '.join(extension['selected_algorithms']) or 'なし'}。旧誤差0.08は従来と同じ暫定閾値で、知覚的成功を保証しない。表は全seedの最終評価、±は母標準偏差（ddof=0）。", ""]
    initial_rows = read(f"constant_{config['steps']}/summary.json")
    lines += ["| 初回の手法 | alignedが初期から改善 | aligned初期の新損失平均 | aligned最終の新損失平均 | baseline最終の新損失平均 |",
              "|---|---:|---:|---:|---:|"]
    for algorithm in config["algorithms"]:
        aligned = [r for r in initial_rows if r["algorithm"] == algorithm and r["profile"] == "balanced_aligned"]
        previous = [read(f"constant_{config['steps']}/balanced_aligned/{algorithm}_{r['seed']}/curve.json")[0]["objective_error"] for r in aligned]
        final = [r["result"]["objective_error"] for r in aligned]
        base = [r["result"]["objective_error"] for r in initial_rows if r["algorithm"] == algorithm and r["profile"] == "baseline"]
        lines.append(f"| {algorithm} | {sum(b < a for a, b in zip(previous, final))}/{len(aligned)} | {np.mean(previous):.5f} | {np.mean(final):.5f} | {np.mean(base):.5f} |")
    lines.append("")
    for directory, cohort in cohorts:
        budget = cohort[0]["budget"]
        lines += [f"### 定数目標・{budget}step/seed", "",
                  "| 手法 | profile | 旧全体誤差 | 新全体損失 | 旧≤0.08 | 持続発声 | 活動20 ms区間率 | 左速度std | 右速度std | 訓練発散 | 評価step/seed |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for algorithm in dict.fromkeys(r["algorithm"] for r in cohort):
            for profile in PROFILES:
                group = [r for r in cohort if r["algorithm"] == algorithm and r["profile"] == profile]
                results, voices = [r["result"] for r in group], [r["voice"] for r in group]
                lines.append(f"| {algorithm} | {profile} | {fmt([r['error'] for r in results])} | {fmt([r['objective_error'] for r in results])} | {sum(r['error'] <= config['threshold'] for r in results)}/{len(group)} | {sum(v['sustained_acoustic'] for v in voices)}/{len(group)} | {fmt([v['active_20ms_fraction'] for v in voices],3)} | {fmt([v['motion']['std'][1] for v in voices],2)} | {fmt([v['motion']['std'][3] for v in voices],2)} | {sum(r['training_numerical_failures'] for r in results)} | {results[0]['evaluation_steps']} |")
        lines += ["", f"![{budget}stepの全seedと平均](../results/issue10/{directory}/comparison_curves.png)", "",
                  "細線が各seed、太線が平均。上段は旧指標、下段は新指標。学習で最適化した指標に加え、共通の評価指標を全条件で計算している。", ""]
    if len(cohorts) == 2:
        def mean_metric(algorithm, profile, budget, metric):
            return np.mean([r["result"][metric] for r in rows if r["algorithm"] == algorithm
                            and r["profile"] == profile and r["budget"] == budget])
        lo, hi = config["steps"], config["extension_steps"]
        lines += [f"TD3 alignedは追加予算で旧誤差平均{mean_metric('TD3', 'balanced_aligned', lo, 'error'):.4f}→{mean_metric('TD3', 'balanced_aligned', hi, 'error'):.4f}、新損失{mean_metric('TD3', 'balanced_aligned', lo, 'objective_error'):.4f}→{mean_metric('TD3', 'balanced_aligned', hi, 'objective_error'):.4f}へ改善した。同予算baselineの旧誤差平均{mean_metric('TD3', 'baseline', hi, 'error'):.4f}も下回るが、seed 1ではbaselineの方が良く、3 seedの平均差だけで優位を確定しない。SAC alignedでは旧誤差平均{mean_metric('SAC', 'balanced_aligned', lo, 'error'):.4f}→{mean_metric('SAC', 'balanced_aligned', hi, 'error'):.4f}と改善が続かず、手法・予算への依存がある。いずれも暫定到達閾値には届いていない。", ""]
    lines += ["### 定数・ランダム・既知制御・直接探索", "",
              "| profile | 比較法 | 旧全体誤差 | 新全体損失 | 探索step/seed |", "|---|---|---:|---:|---:|"]
    for profile in PROFILES:
        for method in ("oracle", "constant", "random", "DE"):
            group = [r for r in baseline if r["profile"] == profile and r["method"] == method]
            steps = group[0].get("search_steps", 0)
            lines.append(f"| {profile} | {method} | {fmt([r['error'] for r in group],6)} | {fmt([r['objective_error'] for r in group],6)} | {steps} |")
    lines += ["", "oracle/constant/randomは各35stepの評価だけで、探索予算0。constantは正規化行動0、randomは各stepで独立一様。oracleの約1e−7は制御encodeのfloat32丸めを含むが、同一配列の固定診断は厳密0。", "",
              "DEは2個の定数をpopsize3×2次元=6候補、初期＋4世代=30episode・1050環境stepで探索し、各profileの実際の割引報酬costを最小化する。新全体を直接最適化するのはalignedだけである。RLは35×2の時変出力が可能で探索次元が異なる。4200stepのRLに対してDEも4200stepへ延長した対照ではない。評価回数や計算秒数も揃えた計算量比較ではない。全候補の曲線と選択制御は [constant_baselines/](../results/issue10/constant_baselines/)。", "",
              f"全学習で最小の旧誤差は {best['result']['error']:.5f}（{best['algorithm']} / {best['profile']} / seed {best['seed']} / {best['budget']}step）。これは最良例の記述であり、手法の一貫した優位性ではない。全seedの値・左右速度std・行動変化MSE・実適用制御の平均/標準偏差は [per_seed.csv](../results/issue10/per_seed.csv)。最終評価の35区間別損失、音声、5系列の制御図、重みは各runディレクトリに保存した。", "",
              "発声はIssue #9の操作的指標（後半0.1–0.35秒、基準帯域RMS比−60 dB以上の20 ms区間が80%以上）。速度stdは全10 MHz積分点の区間統計を平均差も含めて合成する。位置オフセット、源の運動、出力音声、可視性、模倣誤差を区別し、周期性だけで雑音様の音を除外しない。右速度があってもCase Aの音響出力へ右源が寄与するとは限らない。", "",
              "評価用NPZは元の振幅を保持する。`generated.wav` は聴取用にrunごとのpeakを最大1へ収めたfloat32で、波形ごとの音量調整を含む。このWAVの再生音量で条件間の振幅を比較しない。学習・評価・保存誤差にはこの聴取用調整を使用していない。", "",
              "## 4. glideへの移行と観測の限界", "",
              "定数alignedの最終旧誤差≤0.08が同一手法・同一予算で2/3 seed以上となることを、glide学習へ進む事前条件とした。新損失が下がっただけでは移行しない。", ""]
    if not glide["selected_algorithms"]:
        oracle = read("glide_oracle/summary.json")
        lines += [f"条件を満たす手法はなかったためglideのRLは実施しなかった。ε_l=3e7→4.6e7の0.35秒目標を生成し、既知制御のみ旧誤差{oracle['error']:.3g}で再現できることを確認した。このoracleをRL成績には含めない。", "",
                  "![glide既知制御](../results/issue10/glide_oracle/spectrogram.png)", ""]
    else:
        lines += [f"glide学習の対象は {', '.join(glide['selected_algorithms'])}。結果は対応する `glide_*` ディレクトリ。", ""]
    lines += ["現行観測は正規化時刻、前行動2値、現在のx_l,y_l,x_r,y_rの計7次元。遅延線の全履歴、音声フィルタ状態、目標音声の特徴は観測されず、Markov状態を完全には与えていない。固定の一目標で時刻から軌道を覚える設定であり、未知の目標に合わせる目標条件付き模倣とはいえない。今回の比較では観測を変えていない。複数目標では目標特徴と履歴を別条件として検証する必要がある。", "",
              "## 5. 検証・再現・成果物", "",
              "[issue10_validation.txt](issue10_validation.txt) の36テストが成功。旧C++との回帰、RK4・遅延・分割積分、異常終了に加え、oracle/無音/DC/誤周波数の順序、正しい無音・雑音、ゲイン・過大音量・音節時刻、目標マスクの不変性、因果prefix、frameの一度だけの割当て、末尾重み、奇数長窓のpadding診断、割引報酬と全体の一致、baselineの波形と報酬不変、表示設定からの独立性を確認した。", "",
              f"保存成果物も再計算した。{verification['runs']}本、訓練合計{verification['training_steps']:,}step、評価合計{verification['evaluation_steps']:,}step。旧誤差の再計算との差最大{verification['max_legacy_recalculation_difference']:.3g}、alignedの割引returnと全体損失との差最大{verification['max_aligned_return_difference']:.3g}。目標ハッシュ・2自由度・予算・評価回数の一致を [verification.json](../results/issue10/verification.json) に記録。DEの探索9450stepと固定方策の評価等はこの訓練合計へ含めない。", "",
              "物理方程式・RK4・dt=1e−7・遅延、`acoustics.py` と表示関数は基準コミットから変更していない。[provenance.json](../results/issue10/provenance.json) にコードと設定のSHA256・基準commitを保存した。図はε_l, ε_r, p_s, f_e,l, f_e,rを使用。表示は10 MHzへリサンプル、Blackman–Harris、245760/184320、gray_r、[-50,-20] dB、0–10 kHz、端を除いた0.012288–0.331776秒を維持する。解析窓・因果報酬と表示窓は別用途として区別する。", "",
              "```bash", "export MPLCONFIGDIR=/tmp/birdsong-mpl",
              "Deeplearning/.venv/bin/python -m Deeplearning.issue10 all",
              "Deeplearning/.venv/bin/python -m unittest discover -s Deeplearning/tests -v",
              "```", "",
              "段階別に `audit`, `train`, `baselines`, `extend`, `glide`, `report` を指定できる。既存学習は同じconfigなら再利用し、設定不一致の上書きを拒否する。auditとbaselinesは合成データを再生成する。段階を単独実行するときは前提となる成果物を先に作る。`report` は学習を再実行しない。", "",
              "## 6. 判断と残る課題", "",
              "既定設定の固定群では無音優遇が未収束の主因という説明は支持されず、Phase 1の対照報酬を残す判断とした。感度診断では無音と遠い有声音の順位逆転も起き、目標・窓・ゲインが違う条件では再診断が必要。新候補は有声欠落を明示的に罰し時間集計も一致させられるが、旧誤差とのトレードオフがある。良好な模倣が成立したという結論には至っていない。Issueの診断・比較結果を残すことと、模倣学習の主目的の達成を区別する。", "",
              "次に調べる対象は、定数2値探索に対するRLの過大な自由度、部分観測、報酬尺度とSAC温度の関係、各損失成分の独立アブレーションである。旧指標でのseed間再現性を確保してからglideのRLへ進む。候補の音響評価は250–10000 Hzを中心とし、帯域外成分の再現を保証しない。3 seed・一つの0.35秒合成目標であり、生理的妥当性、知覚的類似、実録音・他目標への一般化は示していない。", ""]
    (ROOT/"reports/ISSUE10_REPORT.md").write_text("\n".join(lines))
    print(f"Wrote Issue 10 report: {len(rows)} runs, {successes} final legacy-threshold successes", flush=True)


if __name__ == "__main__":
    main()
