## 現在の目的と実施状況（2026-11-15）

4手法すべてで **Jλ = E_spec + λ E_env、λ=1.0 / 0.5** を比較できるように拡張した。同じ次元・同じλの4手法を1枚へまとめ、旧「1画像1手法」の指定は今回の比較図指定へ置き換える。新しいIssueは作成していない。

**ローカル実装・実行・図の確認・整理は完了。今回の変更は未commit・未push。** 正本は `Deeplearning/results/phase1_constant_020s/` のまま。GitHub上の成果物リンクはユーザーのpushまで前版を示す。既存4096回の54 runは今回の作業開始前にmain_deeplearningへ反映済み。

## 実験と元データの再利用

| 手法 | λ=1.0 | λ=0.5 | 今回の新規学習・探索 |
| --- | --- | --- | --- |
| TD3 | 既存9 run | 既存9 run | なし、保存結果を使用 |
| SAC | 既存9 run | 既存9 run | なし、保存結果を使用 |
| DE | 既存9 run | 新規9 run | 3次元×seed0/1/2、各4096候補 |
| Random Search | 既存9候補プール | 同一9プールを再スコア・最良選択 | 36,864保存候補を再利用、候補再生成0 |

比較は72 profile相当だが、独立した候補列は63 run、新しい候補計算はDEの36,864本のみ。Randomのλ間で候補を重複計上しない。

- Case A、0.20秒、50 kHz、1行動で時間一定。1D=ε_l、2D=ε_l/p_s、3D=ε_l/p_s/f0_l。seed0/1/2。
- 目標 `[4.1e7,3.5e7,3.4e6,3.0e5,4.9e4]`、物理範囲、初期状態、遅延履歴、因果フィルタ、DT=1e-7、既存音響関数は不変。f0_lは外力/開閉操作項でHzではない。
- DEはpopulation8、mutation(.5,1)、recombination=.7、初期乱数生成規則を維持。両λの初期8候補が一致し、差は目的関数のλだけであることを監査した。
- TD3/SACの報酬・構造・ハイパーパラメータ・モデル・音声・原データを変更せず再利用。環境や共有ドライバも変更していない。
- 全手法・両λの**共通評価は E_total=E_spec+E_env**。最小化する重み付きJλと分けて記録する。

## 結果：最終方策／実際の4096番目候補

共通E_total、3 seed平均±標本SD（ddof=1）。直接法の4096番目候補と採用可能な最良解は別である。

| 次元 | λ | TD3 | SAC | DE | Random Search |
| --- | --- | --- | --- | --- | --- |
| 1d | 1.0 | 0.0929707 ± 0.0117 | 0.0854868 ± 0.021 | 0 ± 0 | 0.44104 ± 0.163 |
| 1d | 0.5 | 0.084707 ± 0.018 | 0.0934141 ± 0.00533 | 1.03927e-11 ± 1.8e-11 | 0.44104 ± 0.163 |
| 2d | 1.0 | 0.104559 ± 0.0222 | 0.120369 ± 0.127 | 0.0800564 ± 0.139 | 0.313666 ± 0.0627 |
| 2d | 0.5 | 0.0948123 ± 0.00936 | 0.0808675 ± 0.0209 | 2.4748e-13 ± 2.15e-13 | 0.313666 ± 0.0627 |
| 3d | 1.0 | 0.438944 ± 0.00149 | 0.442125 ± 0.000363 | 0.17655 ± 0.226 | 0.502135 ± 0.0382 |
| 3d | 0.5 | 0.439486 ± 0.00178 | 0.445172 ± 0.000218 | 0.103883 ± 0.119 | 0.502135 ± 0.0382 |

## 結果：Jλによる採用解の共通E_total

直接法は全4096候補からminJλを選択。同値は最初の候補。RLは独立評価した33方策からの補助的な選択で、最終方策と置き換えない。

| 次元 | λ | TD3 | SAC | DE | Random Search |
| --- | --- | --- | --- | --- | --- |
| 1d | 1.0 | 0.0311566 ± 0.0209 | 0.0565761 ± 0.0238 | 0 ± 0 | 0.00562469 ± 0.00801 |
| 1d | 0.5 | 0.0473697 ± 0.00907 | 0.0315746 ± 0.0364 | 1.03927e-11 ± 1.8e-11 | 0.00562469 ± 0.00801 |
| 2d | 1.0 | 0.0176887 ± 0.00936 | 0.105328 ± 0.133 | 0.0800564 ± 0.139 | 0.0258704 ± 0.0181 |
| 2d | 0.5 | 0.0391278 ± 0.025 | 0.0336122 ± 0.00714 | 2.4748e-13 ± 2.15e-13 | 0.0279621 ± 0.0217 |
| 3d | 1.0 | 0.438944 ± 0.00149 | 0.442098 ± 0.00032 | 0.17655 ± 0.226 | 0.166631 ± 0.0254 |
| 3d | 0.5 | 0.439477 ± 0.00179 | 0.445078 ± 5.8e-05 | 0.102568 ± 0.12 | 0.179471 ± 0.0475 |

- DE λ=.5は1D/2Dでほぼ0へ到達。3Dの採用解は平均0.102568、SD0.120183でseed差が残る（各seed 0.0228954 / 0.240807 / 0.0440016）。λ=1の平均0.176550から低下したが、3 seedの小規模な結果であり一律の重み優位は結論しない。
- Random λ=.5の採用解平均は1D 0.00562469、2D 0.0279621、3D 0.179471。候補列と実候補曲線はλ=1と同一。2D/3Dの共通評価による事後最良平均は0.0258704 / 0.166631で、Jλ採用解とは異なる。
- minJλの採用解のE_totalは上昇し得る。今回のbest-so-far図をminE_totalの累積最小に置き換えていない。事後minE_totalは補助列として別記。
- RLの旧 `best/agent.zip` は共通E_totalで選ばれたモデル。今回のJ採用時点について保存済みの行動・物理量・評価値を利用し、保存のない中間ネットワークを復元したとは主張しない。
- DEの高精度到達を客観的に示す一方、λ=1の2D/3Dでは平均最小が別手法になるケースもそのまま報告した。E_spec/E_env、最終と採用解、各seed、進行、制御値と目標差はREPORT/CSVに掲載。

## 誤差線図の正本：24枚

6条件（1D/2D/3D × λ=1/.5）それぞれで、**必ずTD3・SAC・DE・Random Searchの4手法を同じ画像に表示**。異なる次元・λを同じ画像に混ぜない。

```text
figures/learning_curves/{1d,2d,3d}_{lambda1,lambda0p5}_{seed0,mean_sd}.png
figures/best_observed/{1d,2d,3d}_{lambda1,lambda0p5}_{seed0,mean_sd}.png
```

前者は現方策/実候補の生評価、後者はminJλ採用解の共通E_total。seed0と3 seed平均±標本SDは別画像。READMEの6行×4リンクを入口にする。

SciencePlots、軸46pt、目盛30pt、凡例29pt（内側）、線2.8pt、共通色。上部タイトル・注釈・グリッド・閾値線なし。両λで共通軸、平滑化・間引き・頂点省略なし、SD帯は薄く表示。24枚すべてと新しい音響/制御図18枚を目視確認し、文字切れ・凡例と線/帯の重なりを検査した。

## 計算予算・時間・評価回数

- 各profileは4096 RL環境step／4096直接法候補。RLの独立方策評価は33音、直接法の曲線は4096実候補評価であり、best選択の評価機会数は同じではない。
- RLは既存4080 gradient iterationsのまま。環境step・native step・勾配更新を区別。今回の追加gradient updateは0。
- DE新規9 run: 36,864候補、native73,728,000,000 step。CPU 9 workers、並列wall **793.39秒（約13分13秒）**、訓練process wall合計7080.83秒。
- Random再スコア合計1.093658秒。元の候補生成時間/native量はsource_sampling_accountと比較CSVに保持。この時間だけを4096音の新規探索速度と比較しない。
- 新規DE検証36音＋未保存のRandom選択音2音＝追加38音。独立監査の54音をさらに別計上。
- 既存54 runは27 workerで実行済み。今回とは並列度/負荷が異なるため、wall差を手法固有の優劣と解釈しない。次元/手法/λごとの平均±SD・追加評価数はREPORTに一覧化した。

## 検証・削除と保持

- [x] 新DE9 runの4096候補、J=.5による選択、初期候補対応、有限値を検証。
- [x] Random全候補の行動・物理量・音響成分・順序が元データと一致し、候補再生成0を検証。
- [x] 追加監査 **37,819 checks passed**、独立native再生54音の波形差0。元54 runの当時の監査822,369 checksは履歴として保持し、今回の元ファイルhash不変確認と併用。
- [x] 関連7テストpassed。失敗時罰則、J採用解のE_total上昇、同値の扱い、4手法図と削除境界を検証。
- [x] 新図の生成・検証・目視確認後、旧単独手法PNG **72枚**を2つの誤差図ディレクトリから削除。別フォルダへの保管なし。削除サイズ5,526,765 bytes。
- [x] 既存モデル・音声・CSV原数値・生データ・スペクトログラム・物理制御図・Phase0等を保持。元3,392ファイルのhash不変、旧CSV5種類の全既存行/列値不変を確認。
- [x] 正本README/REPORT・図一覧を新24枚に変更。リンク切れと、Git追跡削除が対象72画像だけであることを確認。
- [x] 同じIssue #17の目的・進捗・実験結果を今回の仕様へ更新。Issueは閉じず、commit/pushは実施しない。

## 最初に見るファイル

以下は `Deeplearning/results/phase1_constant_020s/` を基準とする。

- `README.md` → 6条件×4図の相対リンク付き索引。
- `REPORT.md` → 全4手法・両λの最終/採用/事後最良、誤差内訳、進行、物理量、時間と評価回数、考察。
- `comparison_per_seed.csv`, `comparison_aggregate.csv`, `comparison_curves.csv`, `checkpoint_comparison.csv`, `lambda_effects.csv`, `parameter_errors.csv`。
- `direct_weight_extension/manifest.json`, `config.json`, `execution.json`, `audit.json`, `csv_preservation.json`, `completion_validation.json`。
- **削除全72ファイル一覧**: `direct_weight_extension/deleted_single_curve_images.txt`。
- 新DEとRandom選択結果: `runs/constant_{1d,2d,3d}/{DE,random_search}/lambda0p5/seed_{0,1,2}/`。`best_objective`と`best_common_posthoc`を区別し、Randomの共有元は`source_reuse.json`に記録。

[正本READMEの公開先](https://github.com/tatsukawa0353/Birdsong_tatsukawa/blob/main_deeplearning/Deeplearning/results/phase1_constant_020s/README.md) / [REPORTの公開先](https://github.com/tatsukawa0353/Birdsong_tatsukawa/blob/main_deeplearning/Deeplearning/results/phase1_constant_020s/REPORT.md)。**今回の更新ファイルは未pushのため、ユーザーのpushまで前版表示。**

## 再検証・再描画

リポジトリ直下で実行する。既存Python環境を使用する。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue17-extend-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17 audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17_redraw
```

追加実験を再現する全コマンドはREPORTに記載。同名結果を上書きせず、拡張前の54 runを持つ作業コピーでprepare/run/audit/render/目視確認/publishを順に行う。

## 研究上の位置付け

一定制御・少数変数・既知目標の課題は静的最適化に近く、DEが有利でも不自然ではない。3Dの狭い持続発声領域やseed差は残る。時変制御、複数音節、未知目標、学習済み方策の再利用などを両者が公平に比較できる将来課題として設計する。それらでRLがDEより優れるという結論は先取りしない。

既存の旧256回版の履歴整理・Phase0の結果は保持し、今回その整理を再実行していない。今回の主作業はDE λ=.5追加、Random保存候補の再評価、4手法の比較と図の視認性改善である。
