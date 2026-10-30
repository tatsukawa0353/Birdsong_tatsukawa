"""Public methods/synthetic findings and a separate private recording report."""
import hashlib
import json
import os
import subprocess

import numpy as np

from .experiments import write_json
from .model import ROOT


def read(path):
    return json.loads(path.read_text())


def average(values):
    values = np.array(values, dtype=float)
    return f"{values.mean():.5f} ± {values.std(ddof=1):.5f}"


def main():
    public = ROOT/"results/issue9"
    private = ROOT/"private_runs/issue9"
    config = read(ROOT/"configs/issue9.json")
    mapping = read(public/"phonation_map.json")
    neutral = read(public/"neutral_controls.json")
    history = read(private/"historical_diagnostics.json")
    known = read(private/"known_pipeline.json")
    compared = read(private/"comparison.json")
    expected = {(v, c, s) for v in config["variants"] for c in config["conditions"] for s in config["seeds"]}
    actual = {(r["variant"], r["condition"], r["seed"]) for r in compared}
    if actual != expected or len(compared) != len(expected):
        raise ValueError("Comparison is incomplete; finish all configured runs before reporting completion")
    for row in compared:
        if row["training_steps"] != config["steps"] or "training_numerical_failures" not in row:
            raise ValueError("Training budget or numerical-failure logging is incomplete")
    sources = [*ROOT.glob("*.py"), *ROOT.glob("native/*.cpp"), *ROOT.glob("configs/*issue9*.json"),
               *ROOT.glob("tests/test_issue9.py")]
    write_json(public/"provenance.json", dict(
        date_jst="2026-10-30", base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
        configuration=config, privacy="Only synthetic diagnostics are stored in this directory"))
    lines = ["# Issue #9 対応レポート", "", "2026-10-30 JST。対象: [Issue #9](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/9)。", "",
             "発声状態の診断、外力の写像の再設計、固定/可変の同予算比較、制御値の保存と図の修正を実施した。本Issueの診断・実装上の完了条件を満たす確認を行った。良好な模倣の獲得やCase間の普遍的な優劣の確立とは区別する。", "",
             "実録音は許諾確認済みの0.35秒版のみを使用した。原音・録音由来の統計表・図・制御・WAV・学習モデルは `private_runs/` 管理を継続し、本レポートには転載しない。詳細は [ローカル限定レポート](../private_runs/issue9/REPORT.md)。GitHubへの投稿・push・Issueクローズは行っていない。", "",
             "## 確認できたこと", "",
             "- 従来の線形写像では正規化行動0が外力5,024,500になる。基準49,000から大きく離れ、今回の一定制御マップではこの値で持続発声が成立しなかった。これは単に表示レンジが狭いという問題ではない。", "- 波形RMSが非ゼロでもDCオフセットや緩やかな過渡を含み得る。位置・速度の平均と標準偏差、音響帯域のRMS、時間区間別の統計、表示で見えるかを分けて診断する必要があった。", "- Case Bではモデルが右外力を1e7へ上書きしていたが、従来の保存・図は上書き前の値になっていた。共通の制約解決処理と実適用履歴を導入し、この不一致を修正した。", "",
             "## A. 発声・運動・表示の切り分け", "",
             "同じPhase 2環境・フィルタで既知の一定制御を再生し、環境のstep経路と保存制御の再生経路で出力が一致することを確認した。両者は同じC++ APIを使用し、モデル方程式の独立した実装検証ではない。元の9実験についても保存制御を再生し、保存音声と `rtol=1e-10, atol=1e-12` で照合した。音声を書き換えて白い図を隠すような操作はしていない。", "",
             "音声の全体RMS・最大絶対値・数学的に全ゼロか・絶対振幅の小ささに加え、0–0.05、0.05–0.1、0.1–0.2、0.2–0.35、0.1–0.35秒のRMS、平均、AC RMS、250–10000 Hzの帯域RMSを記録した。", "",
             "`x_l,y_l,x_r,y_r` は全10 MHz積分点の平均と標準偏差をWelford法で測定し、区間を結合する際には区間平均の違いも含める。100 Hzの制御端点や位置RMSだけでは周期振動を判定しない。位置の平均はオフセット、標準偏差と速度変動は運動の参考値として分離する。制御が時変の場合に平均値を真の静止平衡と断定しない。", "",
             "診断上の持続発声は、既知のCase A基準に対する帯域RMSが-60 dB以上、後半の20 ms区間の80%以上が同閾値を満たすことと定義した。速度標準偏差1.0を運動の参考閾値とした。これらは実験用の判定値であり、知覚上の無音・生体の開閉・厳密な分岐境界の定義ではない。雑音様でも持続帯域成分があれば除外しない。", "",
             "対数図の後半で-50 dB以上の画素が見えるフレームの割合も別に記録した。音響の有無・源の運動・表示の白さ・模倣誤差をそれぞれ別の項目にした。SACの訓練中の数値発散回数と終了episode数をコールバックで集計し、評価時の失敗と区別する。", "",
             "## B. 外力の役割と範囲", "",
             "`f0` は内部変数名として保持し、研究上の外力 `f_e` に対応付ける。ラビアの開口幅そのもの、音響的な基本周波数F0、左右音源の器官数とは同一視しない。", "",
             "現行コードの式は次のとおり（係数A=1）。", "", "```text", "dx_j/dt = y_j",
             "dy_j/dt = -epsilon_j*x_j - b*y_j - c*x_j^2*y_j - f_e,j",
             "          + p_s + (D*y_j-A)*(p_s-pi_tilde)", "```", "",
             "静止平衡では `epsilon_j*x_j = pi* - f_e,j`、`pi* = sum(alpha_j*x_j)/(1+gamma)`。`K=sum(alpha/epsilon)`, `Q=sum(alpha*f_e/epsilon)` とすると `pi*=-Q/(1+gamma-K)`。今回の範囲では分母が正で、正の外力を増やすと平衡位置は負方向へ移る。これは実装上の符号の確認であり、その向きを生体の閉鎖/開放へ直接読み替えない。`x0,h,M` は現行の微分式から参照されず、実際の開口幅をこの状態値だけから確定できない。", "",
             "位置が変わると非線形減衰 `c*x^2*y` も変わる。p_sはA=1の静止式では相殺されても、速度項の係数に残る。したがってepsilon・p_s・外力の組合せで発声状態を調べた。全領域で単調に抑制するという仮定や、負の外力・未調査領域までの一般化はしていない。", "",
             "### 一定制御マップ", "",
             "A/B/C × epsilon 3値 × p_s 3値 × 外力8値の216条件、各0.35秒・dt=1e-7。左右epsilonは同値。A/Cは左右へ同じ外力、Bは左外力を掃引し右を1e7に固定した。初期0.1秒を除いて判定した。", "",
             "| Case | 外力49,000で持続 | 88,000で持続 | 5,024,500で持続 | 1e7で持続 |", "|---|---:|---:|---:|---:|"]
    for case in "ABC":
        counts = []
        for fe in [49000, 88000, 5024500, 1e7]:
            subset = [r for r in mapping if r["condition"] == case and r["requested_fe"] == fe]
            counts.append(f"{sum(r['sustained_acoustic'] for r in subset)}/{len(subset)}")
        lines.append(f"| {case} | {' | '.join(counts)} |")
    lines += ["", "Case Bでは低い左外力でも全epsilon・p_s条件で持続するとは限らない。マップは既存の可動範囲内の粗い標本であり、発声可能域の完全な境界ではない。", "",
              "![Case A 発声マップ](../results/issue9/phonation_map_A.png)", "", "![Case B 発声マップ](../results/issue9/phonation_map_B.png)", "", "![Case C 発声マップ](../results/issue9/phonation_map_C.png)", "",
              "全数値は [phonation_map.json](../results/issue9/phonation_map.json)、判定定義は [map_config.json](../results/issue9/map_config.json)。", "",
              "### 写像の再設計", "",
              "上限1e7を残し、発声停止側へも到達できる写像を用意した。", "", "```text",
              "u = (action + 1)/2",
              "linear: f_e = 49000 + (10000000-49000)*u",
              "power8: f_e = 49000 + (10000000-49000)*u^8", "```", "",
              "power8の行動0は87,871.09375。マップで調べた88,000に近く、行動中心を発声しやすい側へ移すために選んだ。指数8は生理学的な定数ではなく今回の数値的な設計値。旧範囲の端点は維持し、epsilon・p_sの範囲、RK4・遅延、報酬は変えていない。", "",
              "![外力の写像](../results/issue9/force_mapping.png)", "",
              "実際の行動0の一定制御ではfixedとpower8がA/B/Cとも持続発声となり、linearはA/B/Cとも成立しなかった。詳細は [neutral_controls.json](../results/issue9/neutral_controls.json)。初期行動が必ずゼロになるようにSACを書き換えたわけではなく、学習前方策と探索ノイズの影響は比較実験に含めた。", "",
              "### 同一予算の対照実験", "",
              "fixed・linear・power8 × A/B/C × seed 0/1/2の27本を、SAC各1050環境step、同じ0.35秒目標・報酬・共通ゲイン・評価方法で実行した。評価時の探索ノイズは全て無効。fixedでは発声側の外力49,000を固定し、Bの右1e7は維持する。", "",
              "自由度はfixedがA/B/C=2/3/3、linear・power8が3/4/5。固定対可変は自由度・観測次元も異なるため、外力の影響だけを単独で分離した因果推論ではない。linear対power8は各Caseで同じ自由度・seedの比較。長時間学習で問題を覆うことを避け、予算は旧実験と同じにした。", "",
              "個別・集計の誤差、発声持続率、運動量、区間統計、発散回数、WAV、制御図は [非公開レポート](../private_runs/issue9/REPORT.md) に記載。実録音由来の表を公開レポートへコピーしない。固定は診断の対照であり、外力を恒久的に固定する結論ではない。", "",
              "## C. 保存値と制御図", "",
              "`effective_controls()` をモデル適用・decode・再生で共有し、Case Bの右1e7を必ず反映する。環境は設定要求とモデル適用後の値を別履歴として記録する。NPZの `requested_controls` が要求値、`physical_controls` が実適用値。保存時に行動から推測し直さず、実際の `model.controls` を保存する。", "",
              "元の9実験は要求値を残したうえで制御メタデータを修復し、音声配列を保持した。左右の平均・標準偏差も再生から追加した。全A/B/C制御図は `$\\varepsilon_l$`, `$\\varepsilon_r$`, `$p_s$`, `$f_{e,l}$`, `$f_{e,r}$` の5系列とし、Bの右1e7を表示する。", "",
              "軸ラベル46 pt、目盛38 pt、枠線2、内向き目盛長10、science/ieeeを共有関数で適用。5段の図で読めるよう図寸法を調整した。内部のf0名は変更せず互換性を保つ。公開の合成音の制御図も研究表記へ統一した。", "",
              "スペクトログラムはユーザー指定の10 MHz・Blackman–Harris・245760/184320・gray_r・[-50,-20] dB・両端除外を維持し、表示の正規化を報酬へ持ち込んでいない。", "",
              "## 検証と再実行", "",
              "[issue9_validation.txt](issue9_validation.txt) に26テストの結果。既存回帰に加え、A/B/Cの設定要求と実適用値、固定/可変自由度、power写像の端点・中点・逆変換、全積分点の統計と密サンプルの一致、位置オフセットと運動の区別、雑音様の保持、訓練中の数値発散カウントを検証した。", "",
              "```bash", "export MPLCONFIGDIR=/tmp/birdsong-mpl",
              "Deeplearning/.venv/bin/python -m Deeplearning.issue9 all --local-permission-confirmed",
              "Deeplearning/.venv/bin/python -m unittest discover -s Deeplearning/tests -v", "```", "",
              "部分実行は `map`, `neutral`, `history`, `train`, `report`。設定は [issue9.json](../configs/issue9.json)、コードと設定のハッシュは [provenance.json](../results/issue9/provenance.json)。完了済み学習は同じ設定なら再利用し、異なる設定の上書きを拒否する。履歴修復は既存NPZの制御値・診断メタデータを更新するので、旧要求値を残す。", "",
              "Phase 2にも `force_variant` を追加し、[phase2_fixed.json](../configs/phase2_fixed.json) と [phase2_power8.json](../configs/phase2_power8.json) から選べるようにした。旧phase2.jsonの既定動作は再現用に残した。診断の固定条件を研究全体へ強制していない。", "",
              "## 完了条件と限界", "",
              "1. 保存済み実験の区間統計・全積分点の運動・表示可視性を分離し、既知制御で同じ出力経路を検証した。", "2. 式の符号、216条件のマップ、行動0の検証、固定/可変の同予算比較を残した。", "3. 実適用値を保存・描画し、Bの右固定値と5つの研究表記を回帰テストした。", "4. 公開可能なコード・合成診断と、非公開の録音由来結果を分離した。", "",
              "0.35秒・粗い範囲・3 seedの診断であり、全ての発声開始/停止境界や生理学的な開閉を説明しない。部分観測、振幅校正、声道フィルタ差、報酬設計、学習予算の影響は残る。持続発声が戻ったことと模倣が成功したことを区別し、#6・#7の到達精度と学習効率の検証を継続する。", ""]
    (ROOT/"reports/ISSUE9_REPORT.md").write_text("\n".join(lines))

    previous = os.umask(0o077)
    try:
        lines = ["# Issue #9 ローカル詳細レポート", "", "2026-10-30 JST。録音由来のため研究グループ外へ公開しない。", "",
                 "原音は許諾確認済みの0.35秒版のみ。1秒版は使用していない。元の共通ゲインを全対照実験へ再利用し、保存音声や原音を上書きして結果を改善する操作はしていない。", "",
                 "## 旧Phase 2の9実験", "",
                 "数学的に全ゼロか、帯域成分が基準以下か、速度変動が小さいか、表示が白いかを区別する。区間統計の全値はhistorical_diagnostics.json。", "",
                 "| Case | seed | 全体RMS | 最大絶対値 | 後半AC RMS | 後半帯域対基準dB | 持続 | 源運動 | 白い表示 | 全ゼロ |",
                 "|---|---:|---:|---:|---:|---:|---|---|---|---|"]
        for r in history:
            tail = r["intervals"]["steady_10_35"]
            lines.append(f"| {r['condition']} | {r['seed']} | {r['rms']:.5g} | {r['peak']:.5g} | {tail['ac_rms']:.5g} | {r['late_band_relative_db']:.2f} | {r['sustained_acoustic']} | {r.get('contributing_source_motion')} | {r['display_mostly_white']} | {r['exactly_zero']} |")
        lines += ["", "この診断では旧方策の後半の帯域成分・運動は弱いが、数学的な全ゼロではない。DC・緩やかな過渡がRMSに残り得るため、RMSの非ゼロだけで発声ありとは判定しない。旧方策の模倣誤差と源の抑制は別の項目として扱う。", "",
                  "元のB_0/B_1/B_2の保存右外力は実適用1e7へ修復した。上書き前の要求値はrequested_controlsに保持し、各音声配列のSHA-256をhistorical_diagnostics.jsonへ記録した。修復時の独立再生と旧保存音声の一致をassertで検証した。", "",
                  "## 既知制御の経路検証", "", "| Case | 後半帯域対基準dB | 持続 | 源運動 | 表示が白い | 独立再生一致 |", "|---|---:|---|---|---|---|"]
        for r in known:
            lines.append(f"| {r['condition']} | {r['late_band_relative_db']:.2f} | {r['sustained_acoustic']} | {r['contributing_source_motion']} | {r['display_mostly_white']} | {r['pipeline_replay_identical']} |")
        lines += ["", "known_A/B/Cにスペクトログラム、生成WAV、5制御図、波形を保存した。同じPhase 2環境・同じフィルタで既知制御は持続発声を示し、出力経路だけの一般的な故障という説明を支持しない。", "",
                  "## 27本の同予算比較", "",
                  "SAC、seed 0/1/2、各1050訓練step、350stepごとに探索なし評価。同じ目標・報酬・ゲインを使用。fixedは発声側外力49,000を固定、linearは従来写像、power8は同じ上下限で指数8。Bの右は全構成1e7。", "",
                  "| 構成 | Case | 自由度 | 最終誤差 平均±seed標準偏差 | 持続発声 | 訓練発散合計 |", "|---|---|---:|---:|---:|---:|"]
        for variant in config["variants"]:
            for case in "ABC":
                selected = [r for r in compared if r["variant"] == variant and r["condition"] == case]
                lines.append(f"| {variant} | {case} | {selected[0]['free_controls']} | {average([r['error'] for r in selected])} | {sum(r['diagnosis']['sustained_acoustic'] for r in selected)}/3 | {sum(r['training_numerical_failures'] for r in selected)} |")
        lines += ["", "| 構成 | Case | seed | 初期誤差 | 最終誤差 | 後半帯域対基準dB | 持続 | 源運動 | 訓練終了episode | 訓練発散 |", "|---|---|---:|---:|---:|---:|---|---|---:|---:|"]
        for r in compared:
            d = r["diagnosis"]
            lines.append(f"| {r['variant']} | {r['condition']} | {r['seed']} | {r['initial_error']:.5f} | {r['error']:.5f} | {d['late_band_relative_db']:.2f} | {d['sustained_acoustic']} | {d.get('contributing_source_motion')} | {r['training_episodes']} | {r['training_numerical_failures']} |")
        lines += ["", "### 構成全体の傾向", ""]
        for variant in config["variants"]:
            subset = [r for r in compared if r["variant"] == variant]
            lines.append(f"- {variant}: 持続発声 {sum(r['diagnosis']['sustained_acoustic'] for r in subset)}/9、誤差平均 {np.mean([r['error'] for r in subset]):.5f}、最終誤差0.08以下 {sum(r['error'] <= .08 for r in subset)}/9。")
        lines += ["", "誤差平均の全Case集計は参考値でありCaseの自由度差も含む。3 seedの標準偏差は信頼区間ではない。固定対可変では行動・観測次元が異なる。linear対power8は同じCaseの同じ次元で比較したが、探索分布・方策の学習経路も変わる。", "",
                  "linearも今回は新しい診断コード下で再学習しており、旧9本の方策そのものを対照として流用していない。旧実験との学習結果はビット単位で一致していない（legacy_reproduction.json）。一方、旧制御を固定して再生した音声は前述の許容誤差内で一致した。同予算比較の主表は今回の同じコードによる27本で統一している。", "",
                  "今回、fixed・power8は全9本で持続発声を示したが、最終模倣誤差の改善は一貫しなかった。特にB/Cでは発声の弱いlinearのほうが平均誤差は小さかった。現在の音響誤差は、目標から外れた有声音を出すより抑制したほうが小さくなる場合があると示唆される。表示や外力写像だけで模倣学習まで解決したとは言えず、音響報酬・音量校正・到達可能性の切り分けが次の課題。", "",
                  "固定は学習経路の診断用であり、以後も外力を恒久的に固定する結論ではない。power8は発声側への探索を増やしつつ停止側を残す候補で、指数の最適値や生理学的な制御則を発見したという主張ではない。", "",
                  "## 学習曲線と成果物", "",
                  "![固定](learning_curves_fixed.png)", "", "![従来線形](learning_curves_linear.png)", "", "![power8](learning_curves_power8.png)", "",
                  "experiments/<構成>/<Case_seed>/にsummary.json、curve.json、rollout.npz、agent.zip、diagnostics.json、generated.wav、spectrogram.png、controls.pngを保存した。生成WAVの聴取用音量調整は評価に使用していない。", "",
                  "motion_mean/motion_stdは全積分点からの区間統計、statesは制御端点。physical_controlsは実適用値、requested_controlsは制約上書き前の要求値。全Bのphysical_controls[:,4]が1e7であることを検証した。", "",
                  "## 残る限界", "",
                  "判定は後半0.1–0.35秒、250–10000 Hz、基準比-60 dB、20 ms区間の80%という操作的基準。速度標準偏差1.0も参考閾値で、位置の変動だけで開口・周期性を同定しない。ノイズ様の音は排除していない。", "",
                  "部分観測、録音振幅と声道フィルタの差、短い学習予算、制御可能域や報酬の感度は残る。今回の診断を反映して#6・#7を継続し、モデル到達可能性と学習効率を検討する。鳥の神経学的機構を説明する結果ではない。", ""]
        (private/"REPORT.md").write_text("\n".join(lines))
    finally:
        os.umask(previous)
    print("Wrote public methods report and private experimental report")


if __name__ == "__main__":
    main()
