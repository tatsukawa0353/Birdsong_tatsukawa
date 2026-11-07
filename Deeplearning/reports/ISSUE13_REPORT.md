# Issue #13：一定1変数の再現と一定2変数への拡張

実施日：2026-11-07（JST）。対象：[Issue #13](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/13)。

**一定1変数では、旧結果と共通する108評価点がすべて完全一致した。毎エピソード評価でTD3の低誤差点をより細かく観測・保存できたが、最終方策の悪化も再現した。一定2変数ではDEの最終誤差が最小となり、RLの安定した高精度獲得は確認できなかった。**

新しい成果は [phase1_issue13_20261107](../results/phase1_issue13_20261107/) に保存した。[詳細レポート](../results/phase1_issue13_20261107/REPORT.md)、[全seedの数表](../results/phase1_issue13_20261107/per_seed.csv)、[独立監査](../results/phase1_issue13_20261107/final_audit.json) から元データへ辿れる。旧結果と、前の依頼で作成した修正図・描画コードは保持した。

## 1. 実験条件

実験IDは **reproduction_v1**。[専用設定](../configs/issue13_reproduction.json) を新設し、学習率・探索ノイズ・エントロピー係数等の改善実験は実施していない。

| 条件 | 1試行で決める値 | 保持時間 | 手法・seed | 訓練予算／run |
| --- | --- | --- | --- | --- |
| constant_1d | ε_l | 全0.35秒で一定 | TD3/SAC/Random Search/DE × 0/1/2 | 256エピソード／候補 |
| constant_2d | ε_l、p_s | 両方とも全0.35秒で一定 | 同じ4手法 × 0/1/2 | 256エピソード／候補 |

計24 run。**全条件が鳴管モデルCase A**。constant_2dは既存Task B/Cの時変制御とは異なる1行動・2次元の課題で、左右2音源の比較ではない。

目標は旧Task Aの保存済みモデル内生成音をそのまま複製し、元のNPZとSHA256一致を確認した。実録音は読み込んでいない。目標制御は `(ε_l, ε_r, p_s, f_e,l, f_e,r)=(4.1×10⁷, 3.5×10⁷, 3.4×10⁶, 49000, 49000)`。ε_lの範囲2.5×10⁷〜5.0×10⁷、p_sの範囲2.5×10⁶〜4.0×10⁶、非学習変数は目標値に固定。元のreset、RK4刻み1×10⁻⁷秒、遅延、出力フィルタを変更していない。

主評価は**全0.35秒の対数PSD誤差＋5 ms RMS包絡誤差**で、小さいほどよい。γ=1の報酬和はその負値に一致する。生成音ごとのゲイン・位相調整はしない。balancedは補助診断で、訓練・候補選択の目的関数には使わない。SACの学習にはエントロピー項もあり、全手法の学習目的すべてが同一とは述べない。

両条件のfloat64 Oracleは全17,500サンプルが目標と完全一致し、音響誤差0。float32行動では1変数3.04496×10⁻⁹、2変数1.16347×10⁻⁸だった。[Oracle記録](../results/phase1_issue13_20261107/diagnostics/oracles.json) を確認してから主実験へ進んだ。

## 2. 最終性能

各3 seedの平均±標本標準偏差。全手法256本の訓練用音声生成。RLは最終の決定論的方策、直接法はそれまでの最良候補を返す。

| 条件 | TD3 | SAC | Random Search | Differential Evolution |
| --- | --- | --- | --- | --- |
| 一定1変数 | 0.2193975 ± 0.03009 | 0.3001667 ± 0.02590 | 0.007563262 ± 0.006513 | 3.531329×10⁻¹¹ ± 1.041×10⁻¹¹ |
| 一定2変数 | 0.1756284 ± 0.05486 | 0.2475580 ± 0.1063 | 0.1283039 ± 0.08348 | 0.0007384369 ± 0.001236 |

出典：[集計CSV](../results/phase1_issue13_20261107/aggregate.csv)、[各runの値](../results/phase1_issue13_20261107/per_seed.csv)。

1変数のDEはほぼ完全再現し、Random SearchもRLの最終方策より小さい誤差となった。2変数でもDEの最終誤差が最小で、seed別は約3.98×10⁻⁵、2.17×10⁻³、9.98×10⁻⁶。256候補で残った誤差を、完全収束や知覚的な成功の証明とは扱わない。

2変数ではRandom Search・DEの最終誤差が1変数より増えた一方、TD3・SACの最終平均は1変数より小さかった。**「変数を増やすと全手法で悪化する」とは結論できない。** 同じseed番号でもネットワークの入出力次元や探索軌跡が異なり、単一目標・3 seed・有限予算という比較の限界がある。

## 3. 途中の改善と最終悪化

一定1変数TD3の、初期を含む257評価点の最良と最終を比較する。

| seed | 最良時の訓練試行数 | 最良観測誤差 | 最終誤差 |
| --- | --- | --- | --- |
| 0 | 155 | 0.006563203 | 0.2419812 |
| 1 | 109 | 0.0003791324 | 0.1852449 |
| 2 | 203 | 0.0005689197 | 0.2309664 |

旧結果と共通する9時点×4手法×3 seedの108点は、音響誤差がすべて完全一致した。[点単位の照合](../results/phase1_issue13_20261107/historical_A_shared_checkpoints.csv)。最良値が旧報告より小さいのは、同じ学習過程を毎回評価して従来の記録間隔の間を観測した結果で、学習法を改善した結果ではない。

今回は各評価点の行動・適用制御・誤差内訳・勾配更新回数を保存し、最良時点の `best_observed/agent.zip` と音声NPZ/WAVも保存した。最終はrun直下の `agent.zip`・`rollout.npz` に対応する。保存した最良方策・最終方策の行動は独立に照合済み。ただし、選択した最良方策の評価も同じ固定目標によるもので、未知の目標への汎化を測ったものではない。

評価時点は旧実装に合わせ、試行1〜255では当該試行の勾配更新前、最終256では最終更新後とした。`evaluation_phase` と `gradient_updates` を記録している。悪化の原因を学習率やノイズ等のどれかへ断定する実験ではない。

## 4. 発表用図と元データ

- [一定1変数・seed 0の4曲線](../results/phase1_issue13_20261107/figures/constant_1d_learning_curves.png)
- [一定2変数・seed 0の4曲線](../results/phase1_issue13_20261107/figures/constant_2d_learning_curves.png)
- [描画した全座標CSV](../results/phase1_issue13_20261107/seed0_learning_curve_points.csv)
- [曲線の書式・入力ハッシュ](../results/phase1_issue13_20261107/figures/plot_metadata.json)
- [最終スペクトログラムと音声の対応記録](../results/phase1_issue13_20261107/figures/presentation_spectrograms_metadata.json)

両曲線は同じ表示範囲・SciencePlots字体・軸46 pt・目盛りと凡例30 pt・線幅2.8。各手法1本だけで、平均・帯・他seed線・0.08線・タイトル・図内seed表記はない。凡例はグラフ内右上に置き、線と重ならないことを確認した。横軸 `Number of trials` は訓練エピソード／候補数で、独立評価の回数を加算していない。

発表用最終スペクトログラムはseed 0の各runの最終NPZから8枚生成した。上段Target、下段Generatedで、上側のタイトルは置かない。元runの診断用スペクトログラムも保持している。8 runの具体的な曲線JSON・最終PNG・NPZ・生成WAVへのリンクは [詳細レポートの索引](../results/phase1_issue13_20261107/REPORT.md) を参照。

元データの配置は `results/phase1_issue13_20261107/runs/<condition>/<method>/seed_<n>/`。各runの `config.json`、`curve.json`、`summary.json` は直接読めるJSON、音声と制御の正本はfloat64 NPZ。WAVは元振幅を保持したfloat32で、個別のゲイン正規化やクリップをしていない。スペクトログラムのピーク正規化は表示専用。モデル出力はSI音圧へ校正しておらず、公式SAP評価も未実施。

## 5. 計算量と実時間

| 区分 | 本数／行動数 | 数値積分したモデル時間 |
| --- | --- | --- |
| 主訓練 | 6,144 | 2,150.4秒 |
| 追加評価 | 3,108 | 1,087.8秒 |
| Oracle確認 | 4 | 1.4秒 |
| 目標生成 | 0（旧保存データを複製） | 0秒 |

各runの訓練は256本・89.6秒。RLは初期＋毎試行の257評価で89.95秒/runを追加した。直接法は各候補の目的値とbest-so-farを記録して曲線へ再利用し、初期・最終の独立再生2本で0.7秒/runを追加した。初期ランダム候補は訓練の候補1にも数え、無料の探索候補として追加していない。

訓練実時間のrun別合計は1,942.88秒、追加評価は985.58秒、artifact保存は72.00秒。最大3 workerで並列実行しているため、run合計時間3,014.28秒を経過時間とは呼ばない。manifest作成から全訓練完了の集計まで約1,154.00秒で、準備・コマンド間の確認も含む。監査・発表用画像生成・ソフトウェアテストは主訓練と分離した。[計算量の詳細](../results/phase1_issue13_20261107/execution_summary.json)、[run別会計CSV](../results/phase1_issue13_20261107/compute.csv)。

主訓練・評価とも数値発散0。評価密度、方策選択則、ネットワーク更新・保存のコストが異なるため、256本の訓練予算を揃えたことを全計算量の一致とは説明しない。

## 6. コードと検証

新規追加のみで実装した。

| ファイル | 役割 |
| --- | --- |
| [issue13_env.py](../issue13_env.py) | 既存ValidationEnvの小さなadapter。2変数も1行動・35 block保持にする。元の積分・decode・reset・報酬を継承。 |
| [issue13.py](../issue13.py) | Oracle gate、24 run、毎回評価、最良保存、訓練/評価会計、元成果の保護、計算ソースsnapshot。 |
| [issue13_reproduction.json](../configs/issue13_reproduction.json) | 再現実験の設定。改善条件とは混在させない。 |
| [issue13_report.py](../issue13_report.py) | 元JSONからの曲線・CSV・詳細レポート、最終NPZからの発表用スペクトログラム。 |
| [issue13_audit.py](../issue13_audit.py) | 学習・物理再積分を行わない独立監査。 |
| [test_issue13.py](../tests/test_issue13.py) | Oracle、旧1D等価性、2D保持の等価性、固定値・カウント・失敗・seed継続の8テスト。 |

環境の8テストは全PASS。最終独立監査は**107,130項目すべてPASS**。24 run・曲線6,168点を検査し、保存済み74 NPZ（初期/最終/最良72＋float64 Oracle2）の主音響誤差・balancedを独立再計算した。WAV148本、最終/最良checkpoint24本の行動も一致。既存1,543ファイルと計算ソースsnapshotのSHA256を照合し、不変を確認した。[監査結果](../results/phase1_issue13_20261107/final_audit.json)、[保護検査](../results/phase1_issue13_20261107/preservation_check.json)。

未保存の途中の音声全てを再生成したわけではない。途中はJSONの制御・誤差内訳・更新回数・最良値追跡の整合性を確認し、保存した初期/最終/最良について波形から独立確認した範囲を区別する。

既存のモデル・報酬・旧結果を変更せず、新規フォルダ以外の結果を上書きしていない。実録音の非公開データも参照・転載していない。Gitへのcommit/pushやIssueのクローズは行っていない。

## 7. 再生成と次の判断

以下は既存の新結果を読むだけで、再学習しない。

```bash
Deeplearning/.venv/bin/python -m Deeplearning.issue13_audit Deeplearning/results/phase1_issue13_20261107
Deeplearning/.venv/bin/python -m Deeplearning.issue13_report Deeplearning/results/phase1_issue13_20261107
```

実験を再実行する場合は、別の新しい `--output` を指定し、`issue13 prepare` → `compare --condition constant_1d` → `compare --condition constant_2d` の順に実行する。同じ出力先へのprepare・run上書きは拒否する。

次は、最良方策の選択と最終方策の維持を分けた評価、1項目ずつの学習設定変更、一定2変数での予算依存性を検証する候補がある。今回は学習率・ノイズ・エントロピー等の原因を分離した改善実験をしていないため、どの変更が有効かは未解決。実録音への一般化や左右2音源の優位性も、この結果からは主張しない。
