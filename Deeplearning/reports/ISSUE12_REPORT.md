# Issue #12 診断方法・再現手順

対象: [Issue #12](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/12)。公開するのは録音に依存しない診断コード・合成テスト・一般的な手順だけとする。

**実録音関連の結果・図・判断は `Deeplearning/private_runs/issue12/REPORT.md`、ゼミ準備用のA/B短縮メモは同フォルダの `SEMINAR_MEMO.md` に保存する。** どちらもローカル限定で、GitHub・Issue・PR・コミットメッセージへ転載しない。各文書の冒頭に最初に見るファイルを記す。

## 対象と範囲

Issue #11の既存 `target.npz`、`target_frames.npz`、各runの `rollout.npz`、`activity_frames.json`、設定・診断値を読み、固定済みの波形と実適用制御だけを調べる。原録音を読み直さず、新たなRL学習、モデルでの出音再生成、報酬の採用変更を行わない。

既存 `objectives.py`・`acoustics.py`・SAC/TD3・物理モデル・RK4・遅延・表示関数は変更しない。全入力ファイル集合とSHA256を診断前後で照合し、Issue #11へ書き込まない。入力・出力の親子を含む重なりと、解決後に私有領域外へ出るパスを拒否する。

## A. 活動マスク

同一の目標・共通スケール・250–10000 Hz因果フィルタ・5 ms包絡で、最大包絡の1/3/5/10%を比較する。閾値は全て診断候補。10%を真の無音とせず、背景・残響・弱い声・本来の雑音様音節を手動で確認する欄を設ける。

目標のマスクと包絡を同じ時間軸に表示し、baseline/alignedの固定波形へ同じ診断を適用する。非活動区間がないときの誤発声率はN/Aとし、無音再現の成功とは解釈しない。baselineの学習報酬はこの活動マスクを使っていない。

## B. 固定波形の成分と順序

`BalancedObjective.frame_parts()` をそのまま計算し、生の5項、既存係数、既存frame時間幅を含む寄与を保存する。元の旧全体誤差、balanced全体損失、保存frame_partsとの一致を確認する。

body/weakは目標10%候補、boundaryは切替frame終端の±12.288 msを優先した排他的な診断領域。生PSDと単位質量のq、絶対包絡・相対エネルギー、各領域の時間寄与を対照する。これらは損失解析であり、表示スペクトログラムの設定変更ではない。

固定スコアは以下を別々に比較する。

- 既定1%と3/5/10%マスク。C/J/Mを制限し、包絡・DCは変えない。
- Cだけ、またはC/Jだけに目標の相対帯域包絡の二乗を掛ける時間重み。frame時間重みの再正規化はしない。
- 10%マスク＋C/J時間重み。複合変更として扱い、独立した因果効果を帰属しない。

異なるスコア定義同士の絶対値を改善と呼ばず、同じスコアの下で候補・恒等oracle・無音・DC・音量差・時間ずれ・雑音の順序を調べる。恒等oracleはモデルが実現できるという証明ではない。合成の正しい無音・誤発声・雑音・弱い発声でも同じ診断を行い、弱い実発声の見逃しを確認する。

## C. 実適用制御と応答

実適用5制御の範囲・std・差分・全変動・物理境界率に加え、power8の非線形性を考慮して元の正規化行動の統計も保存する。構造的に固定された制御は、方策の境界滞在とは区別する。Case Bの右外力は保存履歴と照合する。

10 msごとの帯域/AC RMS、包絡、スペクトル重心、形状の時間変化、保存された左右の全積分点統計を共通時間軸へ置く。同時刻と10/20/30 msラグの相関は探索的な記述であり、因果効果やモデル不応答の証明としない。端点の状態変動を全積分点の振動量と取り違えない。

## 実行

リポジトリ直下、既存の専用venvで実行する。

```bash
export MPLCONFIGDIR=/tmp/birdsong-mpl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
Deeplearning/.venv/bin/python -m Deeplearning.issue12 ab
Deeplearning/.venv/bin/python -m Deeplearning.issue12 controls
Deeplearning/.venv/bin/python -m Deeplearning.issue12 report
```

`ab` がA/Bの短縮メモを先に生成する。全段階は `all`、録音を使わない合成だけは `synthetic`。設定は [issue12.json](../configs/issue12.json)。既定の入力は `private_runs/issue11`、出力は `private_runs/issue12`。異なる診断設定や入力を使う場合は新しいprivate出力先を指定する。

録音非依存の合成出力は `results/issue12_synthetic/`。この書出し経路は数学的に波形を生成する専用関数で、任意の入力音声を受け取らない。異なる合成目標同士を混ぜず、tonal/quiet/noiseの同一目標ごとに順位を比較する。

## 検証

[issue12_validation.txt](issue12_validation.txt) に56件の合成テスト・既存回帰の結果を保存した。既定損失と領域寄与の一致、マスク変更と時間重みの作用範囲、無音・雑音・弱音・誤発声・時間位置、qの数値的ゼロ閾値、順位とtie、固定制御と相関、パス制約、入力の内容/ファイル集合の変更検知を含む。

既存回帰に含まれる合成入力の短い学習API異常終了テストは実行するが、実録音を用いる追加のRL実験は行わない。

```bash
Deeplearning/.venv/bin/python -m unittest discover -s Deeplearning/tests -v
```

活動マスク・弱音の時間重み・制御次元/探索の優先順位と、確定事項/仮説の区分はローカルレポートだけに記す。模倣成功、報酬変更の採用、4200stepの十分性、モデル表現限界、左右2音源の普遍的優位性をこの診断だけから結論しない。
