# 2026年11月10日ゼミ：研究進捗の発表用ガイド

作成日：2026-11-01（JST）。対象は既存の公開可能な合成音実験。スライド、切り出し画像、学習結果は今回新規作成していない。

**読み方：全ファイルパスは、このガイドのある `Deeplearning/` からの相対パス。** リンク先は既存ファイル／ディレクトリを指す。本文の数値は既存レポート・保存結果からの転記で、新しい実験や結果の推定ではない。

まず使う資料は、[Phase 0の学習曲線](results/phase0/learning_curves.png)、[Phase 1の1D誤差マップ](results/phase1_validation_20261031/figures/landscape_1d.png)、[Phase 1の学習曲線](results/phase1_validation_20261031/figures/learning_curves.png)、下記Dの数値表。主な根拠は [Phase 1レポート](results/phase1_validation_20261031/REPORT.md) と [考察](results/phase1_validation_20261031/DISCUSSION.md)。図の詳細な所在・優先度はCにまとめた。

## A. 発表概要

| 項目 | 発表での説明案 |
| --- | --- |
| 仮タイトル | **鳴管モデルの音響模倣に向けた制御手法の基礎検証** ― 左右2音源とさえずり多様性の研究に向けて |
| 研究全体の目的 | 鳴管内の左右2音源が、さえずりの多様性に与える影響を調べる。そのために、将来、複数の目標音声を模倣できる制御手法を検討する。これは研究目的であり、今回達成した結論ではない。 |
| 今回の検証範囲 | Phase 0の正弦波追従と、**モデルCase Aに固定したPhase 1**。Phase 1の目標は既知の一定制御から作った0.35秒のモデル内生成音。実録音の成績・図は発表資料に含めない。 |
| 中心となる結論 | **既知制御による再現可能性と実装整合性は確認できた。一方、今回の予算ではTD3・SACの安定した最終性能は確認できなかった。** 一定1変数の直接最適化は良好であり、RLを評価する強い対照になった。 |
| 未検証 | 左右2音源の優位性、複雑な時変目標・複数目標への対応、未知の目標・初期状態への一般化、実録音への一般化、公式SAPによる評価。Oracle一致は物理モデルの正しさやRK4の刻み収束の証明でもない。 |

### 最初に用語を区別する

| 区分 | 意味 | 今回の扱い |
| --- | --- | --- |
| **モデルCase A** | 右側の出力寄与係数α・βを0にする条件。右側の力学状態が消える／停止するという意味ではない。 | Phase 1は全Taskでこのモデル条件を使用。 |
| モデルCase B/C | 音源の寄与・右外力などが異なる別のモデル条件。 | 今回の主比較の対象外。 |
| **Task A** | ε_lだけを0.35秒一定にする、1 episode・1行動の課題。 | 定常1次元。逐次制御RLとは別のバンディット型課題。 |
| **Task B** | ε_lを10 msごとに変更可能。 | 1変数・35行動。直接法の探索次元は35。 |
| **Task C** | ε_lとp_sを10 msごとに変更可能。 | 2変数・35行動。直接法の探索次元は70。 |

音源条件の出典：[モデル調査](reports/model_audit.md)「実装差と音源条件」。Taskの出典：[Phase 1レポート](results/phase1_validation_20261031/REPORT.md)「手法比較」。**Task Cの結果を、左右2音源を使うモデルCase Cの結果とは呼ばない。**

## B. スライド構成案（10枚）

下記は発表の組み立て案。既存画像を再生成する必要はない。部分パネルを説明する場合も、元ファイルはそのまま保持する。

### 1. 研究目的と今回の位置づけ

- **伝える結論：** 左右2音源と多様性の研究に向け、まず音響模倣の制御手法を検証している。
- **載せる図・表：** 研究目的→制御手法の検証→将来の音源比較、という短い文章と、今回の検証範囲。既存画像を無理に追加しない。
- **実際の相対パス：** [reports/model_audit.md](reports/model_audit.md)、[results/phase1_validation_20261031/DISCUSSION.md](results/phase1_validation_20261031/DISCUSSION.md)。文章の根拠として参照する。
- **読む際の注意：** 鳴管は1器官で左右に2音源がある。「鳴管が2個ある」と表現しない。研究目的を達成済みの結果と混同しない。
- **口頭説明：** 「今回は左右2音源の優劣を示す発表ではなく、その比較を可能にする制御法の基礎検証を報告します」。

### 2. Phase 0：正弦波で候補手法を確認

- **伝える結論：** この簡単な追従課題・予算ではTD3、SACがPPOより小さい最終誤差となり、次段階の候補にした。
- **載せる図・表：** Phase 0学習曲線（CのP0-1）とDのPhase 0最終誤差表。
- **実際の相対パス：** [results/phase0/learning_curves.png](results/phase0/learning_curves.png)、[REPORT.md](REPORT.md)「2. Phase 0」、[results/phase0/ranking.json](results/phase0/ranking.json)。
- **読む際の注意：** 横軸は訓練環境step、各手法3本のseed別曲線。平均・標準偏差の図ではない。学習曲線末尾と別の最終評価値は区別する。
- **口頭説明：** 「各4096step・3 seed。目標周波数・振幅を観測できる正弦波課題での暫定選定です。鳴管モデルで同じ順位・収束を保証するものではありません」。

### 3. Phase 1：目標と課題を定義

- **伝える結論：** 正解制御が分かるモデル内目標を用い、一定1変数→時変1変数→時変2変数と段階的に調べた。
- **載せる図・表：** Aの用語表、Dの予算表、目標制御値の短い表。画像より課題の違いを優先。
- **実際の相対パス：** [results/phase1_validation_20261031/manifest.json](results/phase1_validation_20261031/manifest.json)、[results/phase1_validation_20261031/REPORT.md](results/phase1_validation_20261031/REPORT.md)。
- **読む際の注意：** 全Taskで目標は同じ**一定制御による音**。B/Cは制御を時変にできる課題であり、目標そのものが時変制御で生成された実験ではない。全TaskともモデルCase A。
- **口頭説明：** 「ε_lは左側の制御係数、p_sは圧力の制御値です。1変数課題では非学習のp_sを目標生成値に合わせています。全0.35秒の音響誤差を共通評価とし、環境の報酬和もその負値に一致させ、balancedは補助診断にしています」。

### 4. 学習前に誤差分布を調べる

- **伝える結論：** 既知正解とその近傍に低誤差点があることを、粗い探索で確認した。
- **載せる図・表：** 1D誤差マップ（P1-1）を主図とする。2D誤差マップ（P1-2）は補足へ。
- **実際の相対パス：** [results/phase1_validation_20261031/figures/landscape_1d.png](results/phase1_validation_20261031/figures/landscape_1d.png)、[results/phase1_validation_20261031/figures/landscape_2d.png](results/phase1_validation_20261031/figures/landscape_2d.png)。
- **読む際の注意：** 1Dの上段は主音響誤差、下段はbalanced診断値で尺度が異なる。正解点を追加した22候補／140候補のグリッドであり、低誤差点の割合はランダム成功確率ではない。候補間の極小や谷幅は未確定。
- **口頭説明：** 「マップは到達可能性の診断用です。正解点やマップの良い候補を、学習器・最適化器の初期値として渡してはいません」。

### 5. Oracle：実装の整合性を確認

- **伝える結論：** 同じモデル・固定制御・初期状態なら目標を再現できる。学習結果と到達可能性を分けて評価できる。
- **載せる図・表：** Oracleのスペクトログラム（P1-5）とDのOracle表。固定p_sの不一致と解消を一言添える。
- **実際の相対パス：** [results/phase1_validation_20261031/diagnostics/oracle_A/spectrogram.png](results/phase1_validation_20261031/diagnostics/oracle_A/spectrogram.png)、[results/phase1_validation_20261031/diagnostics/oracles.json](results/phase1_validation_20261031/diagnostics/oracles.json)。
- **読む際の注意：** float64で全17500サンプル一致、全区間誤差0。Oracleは既知制御の再生であり学習手法ではない。図の見た目だけで完全一致を判断しない。
- **口頭説明：** 「旧decodeでは1変数制御の固定圧力が目標と違い、正解ε_lでも誤差が残りました。新しい検証環境で固定値を揃えた後、Oracleで整合を確認しました」。

### 6. Task A：最終性能の手法比較

- **伝える結論：** 一定1変数では直接最適化が強く、TD3・SACの最終方策には誤差が残った。
- **載せる図・表：** DのTask A行の4手法比較表を中心に、DEのスペクトログラム（P1-6）を添える。
- **実際の相対パス：** [results/phase1_validation_20261031/REPORT.md](results/phase1_validation_20261031/REPORT.md)「手法比較」の数表、[results/phase1_validation_20261031/per_seed.csv](results/phase1_validation_20261031/per_seed.csv)、[results/phase1_validation_20261031/runs/A/DE/seed_0/spectrogram.png](results/phase1_validation_20261031/runs/A/DE/seed_0/spectrogram.png)。
- **読む際の注意：** 数表は3 seed平均±標本標準偏差。DE図はseed 0の1例。直接法はbest-so-far、RLは最終actorを返し、選択則は異なる。
- **口頭説明：** 「各手法256 episode分の音を生成しています。DE平均誤差は約3.5×10⁻¹¹、ランダム探索は約0.0076。RL固有の優位性は示せていません」。

### 7. TD3：途中の改善と最終悪化を分ける

- **伝える結論：** 途中で低誤差を観測しても、最終方策で維持できていない。
- **載せる図・表：** Phase 1学習曲線（P1-4）の**左上＝Task A・主指標**、Dの途中最良／最終表。余白があれば最終TD3スペクトログラム（P1-7）。
- **実際の相対パス：** [results/phase1_validation_20261031/figures/learning_curves.png](results/phase1_validation_20261031/figures/learning_curves.png)、[results/phase1_validation_20261031/A_diagnosis.json](results/phase1_validation_20261031/A_diagnosis.json)、[results/phase1_validation_20261031/runs/A/TD3/seed_0/spectrogram.png](results/phase1_validation_20261031/runs/A/TD3/seed_0/spectrogram.png)。
- **読む際の注意：** 太線はseed平均、帯は標本標準偏差、細線は各seed。全seedで途中に0.08以下を観測したが、同じ評価回で全seedが0.08以下だったという意味ではない。TD3画像は**最終方策**のもので、途中最良時点の画像ではない。
- **口頭説明：** 「初期・最終を含む9評価点の最良値は記録できていますが、途中の重み・音声は保存していません。最良方策の保存と、その性能の保持が次の課題です」。

### 8. 生成音・制御値・振幅を照合

- **伝える結論：** 周波数構造、実際の制御値、振幅を別々に確認する必要がある。
- **載せる図・表：** TD3最終の振幅包絡（P1-9）を主図とし、制御図（P1-8）はε_l段を指して説明する。縦長の5段図を小さく詰め込まない。DE／TD3スペクトログラムは前スライドの図を参照する。
- **実際の相対パス：** [results/phase1_validation_20261031/runs/A/TD3/seed_0/controls.png](results/phase1_validation_20261031/runs/A/TD3/seed_0/controls.png)、[results/phase1_validation_20261031/runs/A/TD3/seed_0/amplitude_envelope.png](results/phase1_validation_20261031/runs/A/TD3/seed_0/amplitude_envelope.png)。
- **読む際の注意：** Task Aなので制御は時間一定。5段中ε_l以外は固定。正解ε_lの比較線は図に重ねていないため、目標4.1×10⁷と照合して説明する。スペクトログラムは表示専用の個別ピーク正規化を含み、濃淡だけで絶対振幅を比較しない。
- **口頭説明：** 「振幅包絡は目標と生成を元のモデル出力単位で比較しています。出力はSI音圧に校正されたものではありません」。

### 9. Task B/C：制御自由度を増やした結果

- **伝える結論：** 時変1変数・2変数でも今回の予算では十分な模倣精度に至らず、直接法にも誤差が残り、十分な収束を確認できなかった。
- **載せる図・表：** DのTask B/C最終誤差表と、既存学習曲線の**左列中段・下段**。時変制御の具体例はP1-11を補足へ。
- **実際の相対パス：** [results/phase1_validation_20261031/figures/learning_curves.png](results/phase1_validation_20261031/figures/learning_curves.png)、[results/phase1_validation_20261031/DISCUSSION.md](results/phase1_validation_20261031/DISCUSSION.md)。
- **読む際の注意：** B/Cは各128 episodeで、Aの256 episodeと予算が違う。DEは35/70次元に対して個体数8。成績差を制御自由度だけの因果効果と断定しない。全条件の最終値を示し、良いseedだけを選ばない。
- **口頭説明：** 「BにはAのDE解を保持すれば再現できる解があります。それでも、Bの35次元探索やRLが128 episodeでその解を獲得できたわけではありません」。

### 10. まとめと次の検証

- **伝える結論：** 到達可能性は確認できた。次はRLの収束・最良方策の保存と保持、独立評価を詰める。
- **載せる図・表：** Aの中心結論とFの優先順位1〜3を短い表にする。新しい結果図は不要。
- **実際の相対パス：** [results/phase1_validation_20261031/DISCUSSION.md](results/phase1_validation_20261031/DISCUSSION.md)「次に判断すべきこと」、[results/phase1_validation_20261031/REPORT.md](results/phase1_validation_20261031/REPORT.md)「保存物・SAP評価への準備」。
- **読む際の注意：** 今回から実録音への一般化・左右2音源の優位性へ結論を飛ばさない。SAPは予定であり導入済みではない。
- **口頭説明：** 「Phase 1の基礎検証を確実にしてから、複数目標・実際のさえずりへ進み、最終的に左右2音源の寄与を比較したいと考えています」。

**8枚に短縮する場合：** スライド4と5を統合し、スライド8は補足へ移す。Task Aの最終比較と学習途中／最終の区別は残す。2Dマップ・Issue #10の報酬診断・細かな計算資源表は質疑用にする。

## C. 図とデータの索引

### 図・最終数値表の厳選索引

「必須」は本筋を説明する資料、「推奨」は時間があれば加える資料、「補足」は質疑用。個別図は原則seed 0で揃え、統計的な比較は3 seedの数表で行う。

| ID | 優先度 | 用途・読む箇所 | 実際の相対パス |
| --- | --- | --- | --- |
| P0-1 | 必須 | Phase 0学習曲線。横軸step、色別手法、線種別seed。 | [results/phase0/learning_curves.png](results/phase0/learning_curves.png) |
| P0-2 | 補足 | 正弦波の目標／生成比較。TD3 seed 0の例。 | [results/phase0/TD3_0/spectrogram.png](results/phase0/TD3_0/spectrogram.png) |
| P0-3 | 補足 | 正弦波の周波数・振幅制御。目標線の重ね描きはない。 | [results/phase0/TD3_0/controls.png](results/phase0/TD3_0/controls.png) |
| P1-1 | 必須 | 1変数誤差マップ。主指標は上段、正解位置と測定最小点を見る。 | [results/phase1_validation_20261031/figures/landscape_1d.png](results/phase1_validation_20261031/figures/landscape_1d.png) |
| P1-2 | 補足 | 2変数誤差マップ。左は主指標、右はbalanced。橙枠は測定値の下位10%。 | [results/phase1_validation_20261031/figures/landscape_2d.png](results/phase1_validation_20261031/figures/landscape_2d.png) |
| P1-3 | 必須 | **最終音響誤差は既存の表を使用。**「手法比較」からTask別4手法表を引用。 | [results/phase1_validation_20261031/REPORT.md](results/phase1_validation_20261031/REPORT.md) |
| P1-4 | 必須 | 学習曲線。左列が主指標、上／中／下がTask A/B/C。右列balancedは補足。 | [results/phase1_validation_20261031/figures/learning_curves.png](results/phase1_validation_20261031/figures/learning_curves.png) |
| P1-5 | 必須 | Oracleの目標／生成スペクトログラム。再現可能性の確認。 | [results/phase1_validation_20261031/diagnostics/oracle_A/spectrogram.png](results/phase1_validation_20261031/diagnostics/oracle_A/spectrogram.png) |
| P1-6 | 推奨 | Task A・DE seed 0の最終スペクトログラム。直接最適化の再現例。 | [results/phase1_validation_20261031/runs/A/DE/seed_0/spectrogram.png](results/phase1_validation_20261031/runs/A/DE/seed_0/spectrogram.png) |
| P1-7 | 推奨 | Task A・TD3 seed 0の**最終**スペクトログラム。途中最良ではない。 | [results/phase1_validation_20261031/runs/A/TD3/seed_0/spectrogram.png](results/phase1_validation_20261031/runs/A/TD3/seed_0/spectrogram.png) |
| P1-8 | 推奨 | 同TD3の5制御の時間変化。ε_lの目標値とのずれを確認。 | [results/phase1_validation_20261031/runs/A/TD3/seed_0/controls.png](results/phase1_validation_20261031/runs/A/TD3/seed_0/controls.png) |
| P1-9 | 推奨 | 同TD3の目標／生成の5 ms RMS振幅包絡。共通の元単位。 | [results/phase1_validation_20261031/runs/A/TD3/seed_0/amplitude_envelope.png](results/phase1_validation_20261031/runs/A/TD3/seed_0/amplitude_envelope.png) |
| P1-10 | 補足 | DEの包絡比較。TD3と比較する場合も両図の軸目盛を読む。 | [results/phase1_validation_20261031/runs/A/DE/seed_0/amplitude_envelope.png](results/phase1_validation_20261031/runs/A/DE/seed_0/amplitude_envelope.png) |
| P1-11 | 補足 | Task C・SAC seed 0の時変2変数制御例。最良seedの図として選んだものではない。 | [results/phase1_validation_20261031/runs/C/SAC/seed_0/controls.png](results/phase1_validation_20261031/runs/C/SAC/seed_0/controls.png) |
| P1-12 | 補足 | Task A・SAC seed 0の最終スペクトログラム。全手法を視覚比較する場合のみ追加。 | [results/phase1_validation_20261031/runs/A/SAC/seed_0/spectrogram.png](results/phase1_validation_20261031/runs/A/SAC/seed_0/spectrogram.png) |

最終誤差用の新しい棒グラフは作らず、既存レポート表・CSVを使う。Oracle／DE／TD3の各スペクトログラムは上段がTarget、下段がGenerated。ピーク正規化は**表示専用**で、学習・主評価に生成音ごとのゲイン調整を加えたものではない。

Phase 1の表示条件は10 MHzへ表示用再サンプリング、Blackman–Harris、nperseg=245760、noverlap=184320、gray_r、−50〜−20 dB、0〜10 kHz。完全に含まれる窓の中心だけを表示するため、0.35秒音声の表示範囲は0.012288〜0.331776秒。**主評価はこの表示範囲だけではなく全0.35秒**で計算している。

### 数値・設定・元データの所在

| 優先度 | 確認する内容 | 実際の相対パス |
| --- | --- | --- |
| 必須 | Phase 0の条件・最終値・暫定選定の既存集計 | [REPORT.md](REPORT.md)「2. Phase 0」 |
| 推奨 | Phase 0の4096step・seed・評価間隔 | [results/phase0/config.json](results/phase0/config.json) |
| 必須 | Phase 0のseed別初期・最終誤差、到達step | [results/phase0/summary.json](results/phase0/summary.json) |
| 推奨 | Phase 0の選定基準・順位・継続候補 | [results/phase0/ranking.json](results/phase0/ranking.json) |
| 補足 | 正弦波の周波数・振幅追従誤差 | [results/phase0/tracking.csv](results/phase0/tracking.csv) |
| 必須 | Phase 1の条件・目標・範囲・初期状態・報酬定義 | [results/phase1_validation_20261031/manifest.json](results/phase1_validation_20261031/manifest.json) |
| 必須 | Phase 1の全36 run、初期／最終値・計算量 | [results/phase1_validation_20261031/per_seed.csv](results/phase1_validation_20261031/per_seed.csv) |
| 必須 | Oracle、float32/64、旧1変数の不一致の値 | [results/phase1_validation_20261031/diagnostics/oracles.json](results/phase1_validation_20261031/diagnostics/oracles.json) |
| 推奨 | 1Dマップの各候補の全区間音響誤差・balanced | [results/phase1_validation_20261031/diagnostics/landscape_1d.csv](results/phase1_validation_20261031/diagnostics/landscape_1d.csv) |
| 補足 | 2Dマップの各候補の同指標 | [results/phase1_validation_20261031/diagnostics/landscape_2d.csv](results/phase1_validation_20261031/diagnostics/landscape_2d.csv) |
| 必須 | TD3の途中最良値・時点・最終値の補助診断 | [results/phase1_validation_20261031/A_diagnosis.json](results/phase1_validation_20261031/A_diagnosis.json) |
| 推奨 | TD3 seed 0の9評価点の生記録 | [results/phase1_validation_20261031/runs/A/TD3/seed_0/curve.json](results/phase1_validation_20261031/runs/A/TD3/seed_0/curve.json) |
| 推奨 | 同TD3の最終数値・訓練／評価会計 | [results/phase1_validation_20261031/runs/A/TD3/seed_0/summary.json](results/phase1_validation_20261031/runs/A/TD3/seed_0/summary.json) |
| 補足 | 同TD3の元振幅の目標・生成音・適用制御・行動・rate | [results/phase1_validation_20261031/runs/A/TD3/seed_0/rollout.npz](results/phase1_validation_20261031/runs/A/TD3/seed_0/rollout.npz) |
| 補足 | 保存済みの**最終**TD3モデル。途中最良の重みではない | [results/phase1_validation_20261031/runs/A/TD3/seed_0/agent.zip](results/phase1_validation_20261031/runs/A/TD3/seed_0/agent.zip) |
| 補足 | AのDE解をBで保持した追加評価。主比較と区別 | [results/phase1_validation_20261031/cross_task_replay/summary.json](results/phase1_validation_20261031/cross_task_replay/summary.json) |
| 補足 | 全体の計算量と実時間の内訳 | [results/phase1_validation_20261031/execution_summary.json](results/phase1_validation_20261031/execution_summary.json) |
| 補足 | 保存波形・曲線・会計の既存独立監査 | [results/phase1_validation_20261031/final_audit.json](results/phase1_validation_20261031/final_audit.json) |
| 補足 | 過去の合成音報酬診断。今回の主数表へ混ぜない | [reports/ISSUE10_REPORT.md](reports/ISSUE10_REPORT.md) |

元振幅データを探す場合の具体例は、[DEの目標WAV](results/phase1_validation_20261031/runs/A/DE/seed_0/target_raw.wav)、[DEの生成WAV](results/phase1_validation_20261031/runs/A/DE/seed_0/generated_raw.wav)、[DEのfloat64データ](results/phase1_validation_20261031/runs/A/DE/seed_0/rollout.npz)、[形式説明](results/phase1_validation_20261031/runs/A/DE/seed_0/audio_format.json)。WAVはIEEE float32で±1を超えるモデル振幅を保持しており、通常の聴取用正規化音声ではない。定量値の正本はfloat64 NPZで、再生時のクリップを元データの品質と混同しない。

## D. 主な数値結果

以下はすべて既存の集計・保存値。誤差は小さいほどよい。**0.08は暫定的な比較閾値であり、音響的・知覚的な成功を保証しない。**

### D-1. Phase 0：正弦波追従

評価指標は対数PSD距離＋5 ms RMS包絡誤差の**全0.35秒**での和。学習は乱数化グライド、評価は固定1000→2000 Hzグライド、目標周波数・振幅を観測可能。周波数の制御範囲500〜3000 Hz、振幅0〜1、音声48 kHz、10 msごとの35行動。各手法4096訓練step、seed 0/1/2、512step間隔の評価。評価の350環境step/seedは訓練4096stepとは別に記録されている。

| 手法 | 最終音響誤差（平均±標本標準偏差、3 seed） | 0.08以下に到達したseed数 | 訓練＋途中評価の実時間（平均、参考） |
| --- | --- | --- | --- |
| TD3 | 0.04389 ± 0.00487 | 3/3 | 9.37秒 |
| SAC | 0.07081 ± 0.00307 | 3/3 | 14.64秒 |
| PPO | 0.10945 ± 0.00466 | 0/3 | 4.40秒 |

出典：[REPORT.md](REPORT.md)「2. Phase 0」、[config](results/phase0/config.json)、[summary](results/phase0/summary.json)。選定基準は [ranking](results/phase0/ranking.json) にある「最終held-out音響誤差の中央値・少予算での暫定順位」。広い目標分布や鳴管環境での優劣を検証したものではない。実時間は共有ワークステーション上の参考値で、専用機での厳密な速度順位ではない。

図と数表の読み分け：学習曲線は途中評価の推移、最終数表は別の最終評価。例えばTD3 seed 0の曲線末尾は0.044671、最終summaryは0.041344で同値ではない。最終値を図から読み取って数表を書き換えない。根拠は [TD3の曲線記録](results/phase0/TD3_0/curve.json) と [TD3の最終記録](results/phase0/TD3_0/summary.json)。

### D-2. Phase 1：共通の条件と予算

主指標は既存の `acoustics.distance()` による**0.35秒全区間の対数PSD誤差＋5 ms RMS包絡誤差**。目標ピークの共通振幅スケールを用い、生成音ごとのゲイン・位相調整はしない。balancedは独立した診断値で、今回の訓練・直接法の選択には用いていない。パラメータNRMSEは補助指標で、パラメータ完全一致を音響的成功の必須条件とはしない。

目標制御は `(ε_l, ε_r, p_s, f_e,l, f_e,r) = (4.1×10⁷, 3.5×10⁷, 3.4×10⁶, 49000, 49000)`。ε_lの可動範囲は2.5×10⁷〜5.0×10⁷、p_sは2.5×10⁶〜4.0×10⁶。非学習変数は目標生成値へ固定。初期の位置・速度・遅延履歴は0。RK4刻み1×10⁻⁷秒は維持し、同じ出力処理後の50 kHz音声で評価する。

| Task（全てモデルCase A） | 動かす変数と決定回数 | 直接法の探索次元 | 訓練episode／候補数・手法／seed | 訓練環境step・手法／seed | 訓練の数値積分時間・手法／seed |
| --- | --- | --- | --- | --- | --- |
| A | ε_l、1回決定して0.35秒保持 | 1 | 256 | 256 | 89.6秒 |
| B | ε_l、10 msごと35回 | 35 | 128 | 4480 | 44.8秒 |
| C | ε_l・p_s、10 msごと35回 | 70 | 128 | 4480 | 44.8秒 |

各Taskの全4手法がseed 0/1/2で、計36 run。同じTask内では同じ目標・範囲・評価・生成本数。Task間の予算は同一ではない。初期＋8 checkpointの9評価、合計3.15秒の追加積分/runを訓練とは別に計上。Aの評価間隔は32 episode、B/Cは16 episode。

出典：[Phase 1レポート](results/phase1_validation_20261031/REPORT.md)、[manifest](results/phase1_validation_20261031/manifest.json)、[per_seed.csv](results/phase1_validation_20261031/per_seed.csv)。

### D-3. Oracleと最終性能

Oracleは既知制御の再生で、seed学習の成績ではない。

| 確認対象 | 全区間音響誤差 | 意味 |
| --- | --- | --- |
| float64 Oracle（Task A/B/C） | 全て0 | 同じ離散モデル・reset・フィルタで全17500サンプルが一致。 |
| float32 Oracle（Task A/B） | 3.04496×10⁻⁹ | 方策の行動精度に合わせても数値誤差レベル。 |
| float32 Oracle（Task C） | 1.16347×10⁻⁸ | 今回のRL残差を丸めだけで説明する根拠はない。 |
| 旧1変数decodeの固定p_s不一致 | 0.392859 | 固定圧力が目標3.4×10⁶ではなく3.0×10⁶となる整合性の問題。 |

出典：[oracles.json](results/phase1_validation_20261031/diagnostics/oracles.json)、[考察](results/phase1_validation_20261031/DISCUSSION.md)。旧1変数decodeのパラメータNRMSEは学習対象ε_lだけの値で、圧力不一致を含まない。小さいNRMSEを音響的成功の証拠にしない。

次表は主評価の**最終値**、各3 seedの平均±標本標準偏差。予算はD-2のとおり、A=256 episode、B/C=128 episode。

| Task | TD3 | SAC | ランダム探索 | DE（差分進化） |
| --- | --- | --- | --- | --- |
| A | 0.219398 ± 0.0301 | 0.300167 ± 0.0259 | 0.00756326 ± 0.00651 | 3.53133×10⁻¹¹ ± 1.04×10⁻¹¹ |
| B | 0.217312 ± 0.0638 | 0.262281 ± 0.00700 | 0.262957 ± 0.00213 | 0.233397 ± 0.0138 |
| C | 0.287440 ± 0.0213 | 0.253838 ± 0.0123 | 0.337348 ± 0.00347 | 0.298696 ± 0.00954 |

- Task AのDEはほぼ完全再現。DE・ランダム探索はそれぞれ3/3 seedが0.08以下。
- TD3・SACの最終方策は全Taskで0/3 seed。B/Cの直接法も0/3 seed。**今回の条件・予算では十分な模倣精度を確認できなかった**と述べ、RL一般の不可能性やモデル表現力不足とは結論しない。
- CのSACは他手法より最終平均が小さいが、初期平均0.246463から最終0.253838へ改善していない。BのTD3はDEより平均が小さいが、seed間の幅が大きい。単純な優位性の主張にしない。
- 学習曲線はRLでは各評価時点の決定論的actor、直接法ではその時点のbest-so-farを評価する。最終数表はRLの最終actorと直接法の最良候補の比較であり、選択則は異なる。直接法の初期値はランダム候補、RLは未学習の決定論的方策で、同じ初期音ではない。

出典：[REPORTの集計表](results/phase1_validation_20261031/REPORT.md)、[DISCUSSIONの判断](results/phase1_validation_20261031/DISCUSSION.md)。

### D-4. Task A・TD3の途中最良と最終

各seedの256訓練episodeの中で、**保存した9評価点の最良値**と最終評価を比べる。全訓練行動を評価した最小値ではない。

| seed | 最良を観測した訓練episode数 | 最良観測誤差 | 最終誤差（256 episode後） |
| --- | --- | --- | --- |
| 0 | 160 | 0.01858 | 0.24198 |
| 1 | 128 | 0.04857 | 0.18524 |
| 2 | 224 | 0.07736 | 0.23097 |

出典：[A_diagnosis.json](results/phase1_validation_20261031/A_diagnosis.json)、[DISCUSSION](results/phase1_validation_20261031/DISCUSSION.md)。途中の良い値を観測したが、途中重み・音声は保存されていない。現在の最終スペクトログラムや最終モデルを、途中最良時点のものとして提示しない。

補足：AのDE解をBで35回保持した別の再生では、3 seedとも元Aの音声・制御とbitwise一致し、誤差2.90〜4.73×10⁻¹¹だった。Bにも許容解があるという確認で、Bの128候補の探索がその解を得た結果ではない。元Aの探索費用256 episode/seedを要した補助比較で、追加評価は合計3 episode・積分1.05秒。出典：[cross_task_replayの記録](results/phase1_validation_20261031/cross_task_replay/summary.json)。

## E. 発表時の注意点と想定質問

### 共通して守る説明

- **到達可能性・発声成立・模倣精度・学習の安定性を分ける。** Oracleの再現をRLの成功とは呼ばず、音が出ることだけで目標に近いとは言わない。
- **Phase 0、過去のIssue #10、今回のPhase 1を同じ訓練条件として混ぜない。** Phase 0の成功は鳴管環境の成功ではない。
- 今回のRL環境はγ=1で、prefix potential差分の報酬和が負の全区間主音響誤差に一致する。過去のrolling-context報酬やbalanced_alignedの再実験ではない。SACの訓練にはentropy正則化もある。
- [Issue #10レポート](reports/ISSUE10_REPORT.md) は、局所報酬と全区間誤差の順位が逆転し得ること、既定の合成条件では「無音優遇」を主因とする説明を支持しなかったことを示す背景資料。全設定で無音優遇を否定したわけではない。今回の結果表への数値の混入は避ける。
- スペクトログラムの表示用ピーク正規化と評価用の共有スケールを区別する。元振幅・制御図・3 seedの数表と併せて説明する。

### 回答案

| 想定質問 | 簡潔な回答案 | 既存の根拠 |
| --- | --- | --- |
| なぜ簡単なモデルから検証するのか？ | 正解制御が分かる課題で、実装・評価・最適化の問題を切り分けるためです。正弦波で候補を確認し、鳴管モデルではOracleで到達可能性を確認しました。 | [Phase 0を含む既存報告](REPORT.md)、[Phase 1の考察](results/phase1_validation_20261031/DISCUSSION.md) |
| なぜモデルCase Aから開始したのか？ | 右の出力寄与を除く既存の基準モデルで、まず制御法の整合性・到達性・収束を確認する位置づけです。Case Aが最も生物学的に正しい／最も優れるとの比較結果を得たわけではありません。 | [既存モデルの基準](reports/model_audit.md)、[今回の検証範囲](results/phase1_validation_20261031/REPORT.md) |
| RLではなく直接最適化では駄目なのか？ | 直接法は重要な候補です。一定1変数ではDEが明確に良好でした。将来の複数目標・時変制御を見据えてRLも比較していますが、現状ではRLを使う優位性は示せていません。 | [手法比較と限界](results/phase1_validation_20261031/DISCUSSION.md) |
| TD3・SACを選定した根拠は？ | 正弦波でPPOも含め、同じ4096step・3 seedで比較しました。最終held-out誤差の中央値に基づく暫定順位がTD3→SAC→PPOだったため、TD3/SACを継続候補にしました。鳴管モデルでの最適手法と確定したものではありません。 | [ranking.json](results/phase0/ranking.json)、[Phase 0の既存集計](REPORT.md) |
| SAPなどの独立指標は導入しているのか？ | 公式SAPは未実施です。主音響誤差とは別にbalancedを診断値として出していますが、公式SAPスコアではありません。将来の独立評価に備えて元振幅のNPZ/WAVを保持しています。 | [SAP評価への準備](results/phase1_validation_20261031/REPORT.md) |
| 今回から左右2音源の優位性を示せるのか？ | 示せません。Phase 1の全TaskはモデルCase Aで、モデルCase B/Cとの対照比較ではありません。Taskの自由度の差を音源数の効果へ読み替えないようにしています。 | [モデル条件](reports/model_audit.md)、[未検証事項](results/phase1_validation_20261031/DISCUSSION.md) |
| TD3が良くなったのに、なぜ模倣成功と言わないのか？ | 途中の評価値は改善しましたが最終で悪化しました。途中最良の重みは未保存です。到達した低誤差の観測と、再生できる安定した方策獲得を分けています。 | [Aの診断](results/phase1_validation_20261031/A_diagnosis.json) |
| B/Cの成績が悪いのは、再現不能だからでは？ | 少なくともOracleでは再現可能です。BではAのDE解を保持した再生も一致しています。今回の結果だけでは、探索次元・予算・観測・収束の影響を切り分けられていません。 | [Oracle](results/phase1_validation_20261031/diagnostics/oracles.json)、[Bへの補助再生](results/phase1_validation_20261031/cross_task_replay/summary.json) |

## F. 今後の課題（実施済み結果と区別した検証候補）

以下は次の検証案であり、今回実施した作業ではない。

| 優先順位 | 次の検証候補 | 既存結果からの理由・判断する内容 |
| --- | --- | --- |
| 1 | **途中最良方策の重み・制御・音声を保存し、保持と再現性を確認** | TD3は途中で低誤差を観測しても最終で悪化し、途中重みを保存していない。checkpoint保存、方策選択のルールと追加評価費用を明示し、最終actorとの両方を全seedで比較する。 |
| 2 | **制御自由度・探索次元と収束の関係を、予算条件を揃えて調べる** | Aの1次元直接法は良好だが、B/Cの35/70次元探索では誤差が残り、十分な収束を確認できなかった。現状はA/B/Cのepisode予算も異なる。自由度だけの因果効果とせず、同じ生成本数・全区間主指標・計算量会計で確認する。探索／entropy・観測・予算は一度に変更しない。 |
| 3 | **SAPによる独立した評価の方法を検討** | 学習・選択に用いた誤差とは独立した音響評価を検討する。SAP評価だけで知覚的妥当性まで確認したとは扱わない。保持済みの元振幅データを出発点にする。SAPの入力・校正／共通振幅変換・評価条件を定義し、公式SAPと独自特徴量を区別する。新報酬への組込みとは別の検証にする。 |
| 4 | **時変目標、複数目標、未知目標・初期状態への拡張** | 今回の全Taskの目標は一定制御からの単一生成音。時間ごとの行動が可能であることと、時変目標への適応能力を証明したことは別。 |
| 5 | **実録音への適用と、その後の左右音源条件の比較** | Phase 1の到達・収束・独立評価を確認した上で将来の研究へ接続する。音源数と可動自由度・最適化予算の違いを混同しない。録音由来の成果は非公開の管理を維持する。 |

根拠：[DISCUSSIONの未解決事項](results/phase1_validation_20261031/DISCUSSION.md)、[REPORTのSAP・検証範囲](results/phase1_validation_20261031/REPORT.md)。

## 付録：将来の整理候補（今回は変更しない）

| 対象 | 整理候補として記録する理由 | 将来確認すべきこと |
| --- | --- | --- |
| [旧Phase 1](results/phase1/)、[旧追加学習](results/phase1_extended/)、[旧終端直接法](results/phase1_terminal_baselines/)、[Issue #10](results/issue10/)、[今回の基礎検証](results/phase1_validation_20261031/) | 同じPhase 1に見える名前でも、報酬・予算・探索自由度が異なる。発表時に取り違えやすい。 | 削除する重複とはみなさず、将来の一覧に実験世代・目的関数・制御次元・公開範囲を付ける候補。 |
| [DEの目標WAV](results/phase1_validation_20261031/runs/A/DE/seed_0/target_raw.wav) と [TD3の目標WAV](results/phase1_validation_20261031/runs/A/TD3/seed_0/target_raw.wav) | この2ファイルは今回の読み取り確認でSHA256が一致。各runの自己完結性のため同一目標が保存されている。 | 共通目標への参照・保存単位の見直し候補。ただし出典・再現性・既存パスへの依存を確認してから別作業とする。今回は両方保持。 |
| [TD3 seed 0の初期評価](results/phase1_validation_20261031/runs/A/TD3/seed_0/initial/) と [最終評価](results/phase1_validation_20261031/runs/A/TD3/seed_0/) | 同名の図があっても初期／最終の比較という異なる役割がある。似た画像を削除対象にしない。 | 将来、評価時点・seed・Taskが分かる索引の充実を検討。今のファイル名・構成を維持する。 |
| [README](README.md)、[全体REPORT](REPORT.md)、[今回のREPORT](results/phase1_validation_20261031/REPORT.md)、[今回のDISCUSSION](results/phase1_validation_20261031/DISCUSSION.md) | 入口となる文書が複数あり、過去結果と最新条件を読み違えやすい。 | 文書の役割・更新日・過去実験との差を将来の入口案内で明示する候補。今回は元文書を編集しない。 |

録音由来の非公開データ・図・数値は今回の索引・転載の対象外。非公開領域を公開側へ統合する整理は行わない。

## 確認記録

- 掲載したローカル相対パス51件（リンク113参照）はすべて実在確認済み。全リンクはこのガイドの配置場所を基準とする。
- 既存の公開成果1513ファイルについて作業前後のSHA256とファイル集合の一致を確認済み。非公開録音・実行環境・ビルド・キャッシュはこの照合対象から除外し、今回の読み取り・変更対象にしない。
- 新規作成はこのMarkdownガイド1ファイルのみ。既存ファイルの削除・移動・リネーム・上書き、モデル／報酬／条件の変更、RL学習、数値積分、新規図の生成は行わない。
