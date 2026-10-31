"""Private reports for offline Issue 12 diagnosis; no recording result export."""
import csv
import hashlib
from pathlib import Path
import re

import numpy as np

from .issue12 import read, write, table, verify_unchanged
from .phase2 import private_output


def records(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def average(rows, key):
    values = [float(r[key]) for r in rows if r.get(key) not in (None, "", "None")]
    return float(np.mean(values)) if values else None


def fmt(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def aggregate_components(output):
    rows = records(output/"regional_components.csv")
    groups = []
    for budget in (1050, 4200):
        for algorithm in ("TD3", "SAC"):
            for profile in ("baseline", "balanced_aligned"):
                for region in ("body", "weak", "boundary"):
                    group = [r for r in rows if int(r["budget"]) == budget and r["algorithm"] == algorithm
                             and r["profile"] == profile and r["region"] == region and r["variant"] == "default"]
                    if not group:
                        continue
                    values = {k: average(group, k) for k in ("transport", "target_weighted_log", "band_envelope", "missing", "dc", "total")}
                    duration = average(group, "duration_fraction")
                    groups.append(dict(budget=budget, algorithm=algorithm, profile=profile, region=region,
                                       duration_fraction=duration, **values,
                                       dominant_component=max((k for k in values if k != "total"), key=values.get)))
    table(output/"component_summary.csv", groups)
    return groups


def target_region_features(output):
    rows = []
    with np.load(output/"target_analysis.npz", allow_pickle=False) as data:
        for region in ("body", "weak", "boundary"):
            selected = data["regions"] == region
            w = data["weights"][selected]
            if not w.size:
                continue
            rows.append(dict(region=region, duration_fraction=float(w.sum()),
                             mean_power_mass=float(np.average(data["power"][selected].sum(axis=1), weights=w)),
                             mean_envelope=float(np.average(data["envelope"][selected], weights=w)),
                             mean_relative_energy=float(np.average((data["envelope"][selected]/data["envelope"].max())**2, weights=w)),
                             mean_q_mass=float(np.average(data["q"][selected].sum(axis=1), weights=w))))
    table(output/"target_region_features.csv", rows)
    return rows


def paired_rankings(output):
    score_rows = records(output/"fixed_scores.csv")
    variants = list(read(output/"config.json")["variants"])
    results = []
    for algorithm in ("TD3", "SAC"):
        for budget in (1050, 4200):
            for variant in variants:
                wins, ties, changed, total = 0, 0, 0, 0
                for b in score_rows:
                    if b["algorithm"] != algorithm or int(b["budget"]) != budget or b["profile"] != "baseline":
                        continue
                    a = next(r for r in score_rows if r["algorithm"] == algorithm and int(r["budget"]) == budget
                             and r["condition"] == b["condition"] and r["seed"] == b["seed"] and r["profile"] == "balanced_aligned")
                    difference = float(a[variant])-float(b[variant])
                    old = float(a["default"])-float(b["default"])
                    wins += difference < -1e-12
                    ties += abs(difference) <= 1e-12
                    changed += old*difference < 0 and abs(old) > 1e-12 and abs(difference) > 1e-12
                    total += 1
                if total:
                    results.append(dict(algorithm=algorithm, budget=budget, variant=variant,
                                        aligned_wins=wins, ties=ties, pairs=total, winner_changes_from_default=changed))
    table(output/"paired_rank_summary.csv", results)
    return results


def seminar_memo(output):
    output = private_output(output)
    masks = read(output/"masks.json")
    components = aggregate_components(output)
    ranking = paired_rankings(output)
    target_features = target_region_features(output)
    check = read(output/"score_verification.json")
    lines = ["# 11月10日ゼミ準備：Issue #12 A/B内部メモ", "", "2026-10-31 JST。録音由来の内容はローカル限定。再学習・報酬採用は行っていない。", "",
             "## 最初に見るもの", "",
             "1. [活動マスクと包絡](activity_masks.png) — 同一目標・同一特徴の1/3/5/10%。",
             "2. [損失成分の時間変化](loss_components_A.png) — 成績で選ばないTD3/4200/Case A/seed 0の両固定波形。B/Cも同名図。",
             "3. [生PSDとqの対照](spectral_normalization_A.png) — body/weak候補の中央値frameと絶対/相対エネルギー。",
             "4. [手動確認欄](manual_review.md) — 背景・残響・弱声・雑音様音節を自動で断定しない。",
             "5. [成分集計](component_summary.csv) と [順位集計](paired_rank_summary.csv) — 既定スコアの時間寄与と同一スコア内の順序。", "",
             "## A. 目標マスク", "", "| 閾値 | 活動時間率 | 低エネルギー秒 |", "|---|---:|---:|"]
    for row in masks:
        lines.append(f"| {row['ratio']:.0%} | {row['active_fraction']:.6f} | {row['low_energy_duration_s']:.6f} |")
    lines += ["", "閾値は診断候補で、10%も真の無音ラベルではない。1%で非活動がなければ誤発声率はN/Aとなり、無音を正しく再現したという推論はできない。候補マスクはbaseline/alignedの両固定波形へ同じように適用した。baselineの学習報酬はこのマスクを使っていない。", "",
              "### 生のパワーと正規化質量", "",
              "以下の領域定義は次節と同じ。条件付き時間平均であり、損失寄与の総和ではない。Pは共通振幅スケール下の特徴power。qの総和1はフレーム正規化の定義で、弱い声・背景が同じ意味になることや、損失全体が音量を無視することを意味しない。", "",
              "| 領域 | 平均P総量 | 平均帯域包絡 | 平均相対エネルギーg | 平均q総量 |", "|---|---:|---:|---:|---:|"]
    for r in target_features:
        lines.append(f"| {r['region']} | {r['mean_power_mass']:.7g} | {r['mean_envelope']:.7g} | {r['mean_relative_energy']:.7g} | {r['mean_q_mass']:.7g} |")
    lines += ["", "power総量が1e−20以下なら既存実装はq=0とする。qの音量不変性はその数値的ゼロ閾値をまたがない範囲の性質であり、極小の減衰尾部まで無条件には成り立たない。", "",
              "## B. 既定スコアの弱い区間に何が寄与するか", "",
              "body/weakは目標10%候補、boundaryはマスクが切り替わったframe終端の±12.288 msを優先して割当てた排他的領域。境界は自動候補で、生物学的な音節区分ではない。以下はweakのみ、4200stepの各Case・seedを合わせた9波形の平均。値は係数とframe時間重みを含む全0.35秒損失への寄与で、領域平均損失と取り違えない。body/boundaryを含む全表はcomponent_summary.csv。", "",
              "| 手法 | 固定波形 | 領域 | 時間率 | shape C | log 0.25J | envelope A | missing 2M | DC 0.1D | 合計 | 最大成分 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in components:
        if r["budget"] != 4200 or r["region"] != "weak":
            continue
        lines.append(f"| {r['algorithm']} | {r['profile']} | {r['region']} | {r['duration_fraction']:.3f} | {r['transport']:.5f} | {r['target_weighted_log']:.5f} | {r['band_envelope']:.5f} | {r['missing']:.5f} | {r['dc']:.5f} | {r['total']:.5f} | {r['dominant_component']} |")
    lines += ["", "C/Jのフレーム正規化は単位質量のqを使い、絶対振幅はAに残る。Mは有声候補での過小出力だけを罰し、過大出力へは直接作用しない。したがってC/Jの正規化を弱めたときの影響と、M/Aが支配しているときの影響を分けて読む。", "",
              "### 同じ固定スコアの下でのペア順序", "",
              "同一手法・Case・seed・予算のbaseline/alignedを比較。数値が異なるスコア定義同士の大小を『改善』とは呼ばない。下表は4200stepの9ペア中、aligned波形のスコアが小さい数と、既定スコアから勝者が変わった数。", "",
              "| 手法 | 既定でalignedが小さい | mask3/5/10%の勝者逆転数 | C/CJ時間重みの勝者逆転数 | 複合変更の勝者逆転数 |", "|---|---:|---|---|---:|"]
    for algorithm in ("TD3", "SAC"):
        group = {r["variant"]: r for r in ranking if r["budget"] == 4200 and r["algorithm"] == algorithm}
        default = group["default"]
        masks_changed = "/".join(str(group[k]["winner_changes_from_default"]) for k in ("mask_03", "mask_05", "mask_10"))
        energy_changed = "/".join(str(group[k]["winner_changes_from_default"]) for k in ("transport_energy", "spectral_energy"))
        lines.append(f"| {algorithm} | {default['aligned_wins']}/{default['pairs']} | {masks_changed} | {energy_changed} | {group['mask_10_spectral_energy']['winner_changes_from_default']} |")
    lines += ["", f"既存{check['runs']}波形の旧全体誤差・既定balanced・保存frame_partsを再計算。最大の保存スコアとの差={check['maximum_residual']:.3g}。元スコアや音声は変更していない。oracle等の比較用固定対照は同じ目標から作ったオフライン対照で、モデルがその波形を生成可能という証明ではない。", "",
              "### ここで確定しないこと", "",
              "弱い成分を全て背景とみなすこと、正規化を外すだけでRLが改善すること、1%マスクが両報酬の未収束の唯一の原因であることは確定しない。合成の弱い実発声・正しい無音・誤発声・雑音・時刻ずれでも診断し、高閾値やエネルギー重みで弱音を軽視する危険を残す。", ""]
    interpretation = output/"interpretation.md"
    if interpretation.exists():
        lines += ["## 診断後の解釈と優先順位", "", interpretation.read_text(), ""]
    else:
        lines += ["## 目視・追加整理欄", "", "manual_review.mdへ根拠と未確定点を記入する。Cは保存制御の別解析として行い、相関を因果効果とは解釈しない。", ""]
    (output/"SEMINAR_MEMO.md").write_text("\n".join(lines))


def main(source, output):
    source, output = private_output(source), private_output(output)
    seminar_memo(output)
    components = aggregate_components(output)
    ranking = paired_rankings(output)
    scores = records(output/"fixed_scores.csv")
    config = read(output/"config.json")
    responses = records(output/"response_summary.csv") if (output/"response_summary.csv").exists() else []
    third_item = ("3. [制御×包絡の対照](TD3_4200/A/baseline/seed_0/controls_and_audio.png) — 実適用5制御、帯域包絡/RMS、左右速度統計。"
                  if responses else "3. [生PSDとqの対照](spectral_normalization_A.png) — Cは後回しにできるA/B段階の図。")
    lines = ["# Issue #12 ローカル診断レポート", "", "2026-10-31 JST。[Issue #12](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/12)。実録音由来の数値・図・解釈は全てローカル限定。", "",
             "## まず見るべき5個", "",
             "1. [活動マスク比較](activity_masks.png) — 包絡と1/3/5/10%候補を同じ時間軸に表示。",
             "2. [項目別損失の時間変化](loss_components_A.png) — 代表TD3/4200/Case A/seed 0の両報酬の固定波形。",
             third_item,
             "4. [ゼミ向けA/B短縮メモ](SEMINAR_MEMO.md) — 2026年11月10日準備用。",
             "5. [生PSDとqの対照](spectral_normalization_A.png) と [手動確認欄](manual_review.md) — 正規化と音節ラベルの区別。", "",
             f"Issue #11の{len(scores)}本の保存波形・制御をオフライン解析した。新たなRL学習、出音の再生成、報酬の採用・係数変更、SAC/TD3・物理式・RK4・遅延・表示関数の変更は行っていない。原録音を読み直さず、Issue #11に保存済みの目標配列だけを使用した。", ""]
    notes = output/"interpretation.md"
    if notes.exists():
        lines += ["## 確認できたことと次の優先順位", "", notes.read_text(), ""]
    else:
        lines += ["## 判断の範囲", "", "閾値別・成分別の数値をSEMINAR_MEMOへ整理した。ラベルの妥当性と因果関係は別に確認する。", ""]
    lines += ["## A/Bの方法と成果物", "",
              "同一目標ピークの共有スケール、250–10000 Hzの因果6次Butterworth、先頭値の定常状態から初期化、50 kHzでBlackman–Harris窓1229sample・hop307sample、因果5 ms RMSの新規区間平均を使う。報酬用の解析窓は左ゼロ補完のみ。frame終端58点と新規時間幅weightsをそのまま使い、末尾1sampleの重みも維持する。これは既定の表示スペクトログラムとは用途が違う。", "",
              "1/3/5/10%はいずれも目標包絡最大の振幅比閾値。生成の活動判定にも同じ目標由来の絶対閾値を使用する。雑音様の目標を周期性の低さだけで除外しない。body/weakは10%候補、boundaryは候補マスク変化点±12.288msで、3領域は重複せず時間重み合計1。真の音節ラベルとして使わない。", "",
              "元のframe_partsをコピーして、マスク変更時にはC/J/Mだけを候補マスクで制限する。元のマスク以上の閾値だけを扱うため、元から非活動のframeを無理に復元していない。AとDCは不変。エネルギー重みg=(E_target/max E_target)^2を指定したCのみ、またはC/Jだけへ適用し、frame時間重みの再正規化はしない。係数は1/0.25/1/2/0.1を維持する。", "",
              "| 固定スコア | マスク | gをかける項 | 比較の意味 |", "|---|---|---|---|"]
    for name, settings in config["variants"].items():
        meaning = "複合変更・独立効果は帰属しない" if settings["ratio"] != .01 and settings["energy_terms"] else "一要因のオフライン比較"
        lines.append(f"| {name} | {settings['ratio']:.0%} | {', '.join(settings['energy_terms']) or 'なし'} | {meaning} |")
    lines += ["", "異なるスコア定義同士の絶対値の大小から報酬改善とは言わない。各手法・予算・Caseごとに6固定波形と同じ目標のoracle/無音/DC/半音量/20ms遅延/RMS一致雑音を同一スコアで並べる。oracleは恒等比較の下限で、モデル内の到達可能性ではない。順位・逆転は [rankings.csv](rankings.csv)、[rank_reversals.json](rank_reversals.json)、同seedペアは [paired_rank_summary.csv](paired_rank_summary.csv)。", "",
              "全frameの生の損失項、係数×時間重みの寄与、全variantのparts/contributions、Pとqは各runのframe_components.csv/features.npz。全runの領域別寄与は [regional_components.csv](regional_components.csv)、集計は [component_summary.csv](component_summary.csv)。全スコアは [fixed_scores.csv](fixed_scores.csv)、全閾値の欠落・誤発声は [activity_comparison.csv](activity_comparison.csv)。旧全体誤差と既定balancedは別列で不変に保持した。", "",
              "### 代表図", "", "![活動マスク](activity_masks.png)", "", "![項目別損失](loss_components_A.png)", "", "![生PSDと相対形状](spectral_normalization_A.png)", "",
              "B/Cも loss_components_B/C.png と spectral_normalization_B/C.png に保存。代表例は成績を見ず、TD3・4200step・seed 0に固定した。P/q図は損失解析の周波数別曲線であり、既定のスペクトログラムを設定変更したものではない。", "",
              "## C. 保存制御と音響の時間変化", ""]
    if responses:
        lines += [f"全{len(responses)}本の実適用5制御と同じ10 msグリッドの出音を確認した。Case Bの右外力1e7は保存値と照合済み。構造上固定の制御は『学習が境界へ張り付いた』という集計から除外する。", "",
                  "物理範囲に対するspan/std/全変動/差分/境界1%滞在率に加え、power8の非線形性を考慮して元の正規化行動のspan/std/差分/境界率も併記する。『ほぼ一定』の操作的旗は全可動制御で行動span≤0.05かつ物理span≤全範囲1%の場合だけ。小さな物理変動でも非線形な発声変化を起こし得るため、この旗を生理的な無変化とは呼ばない。", "",
                  "| 手法 | 予算 | 波形 | ほぼ一定のrun | 可動行動最大span平均 | 生成band RMS CV平均 | 生成centroid std [Hz]平均 |", "|---|---:|---|---:|---:|---:|---:|"]
        for algorithm in ("TD3", "SAC"):
            for budget in (1050, 4200):
                for profile in ("baseline", "balanced_aligned"):
                    group = [r for r in responses if r["algorithm"] == algorithm and int(r["budget"]) == budget and r["profile"] == profile]
                    if group:
                        lines.append(f"| {algorithm} | {budget} | {profile} | {sum(r['near_constant_controls']=='True' for r in group)}/{len(group)} | {average(group,'maximum_free_action_span'):.4f} | {average(group,'band_rms_cv'):.4f} | {average(group,'centroid_std_hz'):.2f} |")
        target_response = read(output/"target_response_summary.json")
        lines += ["", f"同じ診断での目標band RMS CV={target_response['band_rms_cv']:.4f}、centroid std={target_response['centroid_std_hz']:.2f}Hz。centroidはF0ではなく、弱いframeのq正規化にも影響される。RMS CVは平均が極小なら大きくなり得るため、全runの平均RMS・元の持続判定も合わせて読む。", "",
                  "相関は同時刻と0/10/20/30ms後の出音に対して算出し、全ラグを保存した。各run35点で時間自己相関があり、最大相関の探索も因果性や有意性の証明ではない。一定系列の相関はN/Aにする。振幅が一定でも周波数・位相・波形が変わることがあり、RMSだけでモデルが応答しないとは断定しない。閉ループ方策の制御と音響の共変動は、独立した入力介入実験ではない。", "",
                  "全物理・行動統計は [control_statistics.csv](control_statistics.csv)、ラグ別は [control_correlations.csv](control_correlations.csv)、音響変動と元の持続・速度は [response_summary.csv](response_summary.csv)。端点状態と、保存された10MHz積分点の後半のpooled mean/std/RMSは [state_statistics.csv](state_statistics.csv) に分離し、元のdiagnostics.jsonとも照合した。各runに controls_and_audio.png、audio_response.csv、control_statistics.csv、control_correlations.csvを保存。", "",
                  "### 制御×音響の代表図", "", "![実適用制御と音響](TD3_4200/A/baseline/seed_0/controls_and_audio.png)", "",
                  "[同じ例のaligned](TD3_4200/A/balanced_aligned/seed_0/controls_and_audio.png)、[Case B baseline](TD3_4200/B/baseline/seed_0/controls_and_audio.png)、[Case B aligned](TD3_4200/B/balanced_aligned/seed_0/controls_and_audio.png)、[Case C baseline](TD3_4200/C/baseline/seed_0/controls_and_audio.png)、[Case C aligned](TD3_4200/C/balanced_aligned/seed_0/controls_and_audio.png)。", ""]
    else:
        lines += ["Cは未実施。A/Bメモと固定スコア検証を優先して完了した段階のレポート。", ""]
    lines += ["左右x/yの端点、全積分点のmotion_mean/std、発声持続の診断は元の保存値を保持した。Case Bの強制外力値から厳密な振動停止や生体の閉鎖を確定しない。Phase 1は既知制御でモデル自身の音声を作る2制御課題、Phase 2は未知の実録音・3/4/5制御・フィルタとゲインの差を含む。時刻/前行動/現在状態しか観測せず、遅延履歴と目標特徴は未観測。Phase 1の未収束は学習上の問題を残すが、Phase 2のモデル表現限界を確定する証拠ではない。", "",
              "## 検証と再現", "",
              "[score_verification.json](score_verification.json) で旧全体誤差・既定balanced・保存frame_partsを照合。各領域の時間重み付き成分を合計すると既定全体損失へ戻る。元Issue #11の全ファイル集合とSHA256、元の物理・報酬・表示コードの前後一致は [source_integrity.json](source_integrity.json)。既存の学習重みや音声も含め、元フォルダへ書き込んでいない。", "",
              "録音を使わないゼロ・有声・雑音・弱音・誤発声・タイミング・マスク差のテストと既存回帰56件が成功した。既存回帰には合成入力の短い学習API異常終了テストがあり、本Issueの追加RL実験とは区別する。公開可能な合成固定波形と診断表は `Deeplearning/results/issue12_synthetic/`、合成テストログは `Deeplearning/reports/issue12_validation.txt`。高い閾値・エネルギー重みで弱い実発声の欠落が軽く採点され得ることも説明に含める。", "",
              "```bash", "export MPLCONFIGDIR=/tmp/birdsong-mpl", "export OPENBLAS_NUM_THREADS=1", "export OMP_NUM_THREADS=1",
              "Deeplearning/.venv/bin/python -m Deeplearning.issue12 ab",
              "Deeplearning/.venv/bin/python -m Deeplearning.issue12 controls",
              "Deeplearning/.venv/bin/python -m Deeplearning.issue12 report", "```", "",
              "全段階はall、合成だけはsynthetic。入力はprivate_runs/issue11、出力は別のprivate_runs/issue12。親子を含む入力・出力の重なりと、解決後に私有領域外へ出るシンボリックリンクを拒否する。元のスナップショットと設定が変わった場合は新しい出力先を使う。再実行は診断出力だけを更新する。表示スペクトログラムの10 MHz/245760/184320/Blackman–Harris/gray_r/[-50,-20]dB/端除外は変更していない。", ""]
    text = "\n".join(lines)
    (output/"REPORT.md").write_text(text)
    missing = [link for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text)
               if "://" not in link and not (output/link).exists()]
    if missing:
        raise ValueError("Report links missing diagnostic artifacts")
    write(output/"analysis_provenance.json", {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                               for name in ("issue12.py", "issue12_report.py", "offline_diagnostics.py", "configs/issue12.json")})
    verify_unchanged(source, output)
    print("Wrote Issue 12 private report and seminar memo; original Issue 11 files unchanged.", flush=True)
