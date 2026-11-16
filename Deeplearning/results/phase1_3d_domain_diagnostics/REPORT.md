# Issue18 — Case A・3Dの発声領域と制限探索の診断

既存Issue17の数値・モデル・音声を対照として読み取り専用で利用。今回の新規学習はλ=1、3D、4手法×seed0/1/2の12 runだけ。全て4096候補。目標・物理モデル・報酬・ネットワーク・optimizer・seed・CPU条件を維持し、f0_lの行動から物理値への範囲だけを変更した。λ=.5は追加せず、範囲効果と重み効果を分離する。

## 事前の物理診断と範囲の決定

Oracleのε_l/psを固定した全域f0_l sweep、下限付近の線形/対数格子、ε_l×f0_lとps×f0_lの断面、全域ε_l/psの3×3 joint anchorを計3388点で評価。新規最適化ではなく、全ての失敗・非発声点を保持した。[事前計画](diagnostics/plan.json)と[全点](diagnostics/domain_grid.csv)。

発声proxyは既存と同じ `std(最後の1000 sample)/Target peak > .05`、数値失敗なし。最後の20 msのAC標準偏差であり、生理学的有声・無声や全区間RMSと同一ではない。
観測した最大発声f0_l=476950。事前規則 `ceil(1.5*最大発声f0/1e5)*1e5` により上限=800000を追試前に固定。E_totalや目標近傍の誤差を上限決定に使わず、50%の安全余裕と丸めを加えた。[決定記録](domain_decision.json)。

旧f0_l範囲[49000,10000000]→新[49000,800000]。ε_l=[2.5e7,5e7]、ps=[2.5e6,4e6]は維持。目標f0_l=300000は新範囲の約33.4%位置で、目標値中心の狭い窓ではない。反面、既知目標のε_l/psを用いる診断を含むため、未知目標への汎化を証明する設計ではない。断面と9 joint anchorの調査であり、3D全域でこれ以上発声しないことを証明していない。非一様格子の702/3388発声比率を物理体積の確率と誤記しない。

同じ正規化行動幅・TD3 noise=.1でも物理f0_lの探索幅は約13.25倍縮む。これは範囲制約の帰結で、同じ物理探索ノイズを維持した実験ではない。右側係数はCase Aの固定値のまま。f0_lは外力/開閉項でHzではない。元コードを改変せず、spawnした追試worker内だけDomainEnvを供給した。元runのsource hashと追加adapter hashを両方保存。

## 既存履歴からの説明と限界

| 手法 | λ | seed | 記録 | 観測数 | 発声数 | minJ時点 | その共通誤差 | 上限5%域の比率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TD3 | 1.0 | 0 | training_samples | 4096 | 1 | 7 | 0.33274 | 0.627 |
| TD3 | 1.0 | 0 | independent_policy_evaluations | 33 | 0 | 4096 | 0.440233 | 0.758 |
| TD3 | 1.0 | 1 | training_samples | 4096 | 0 | 3275 | 0.438153 | 0.623 |
| TD3 | 1.0 | 1 | independent_policy_evaluations | 33 | 0 | 4096 | 0.439289 | 0.758 |
| TD3 | 1.0 | 2 | training_samples | 4096 | 0 | 2389 | 0.437212 | 0.630 |
| TD3 | 1.0 | 2 | independent_policy_evaluations | 33 | 0 | 4096 | 0.437309 | 0.758 |
| TD3 | 0.5 | 0 | training_samples | 4096 | 1 | 7 | 0.33274 | 0.577 |
| TD3 | 0.5 | 0 | independent_policy_evaluations | 33 | 0 | 3712 | 0.438245 | 0.697 |
| TD3 | 0.5 | 1 | training_samples | 4096 | 0 | 4036 | 0.438654 | 0.631 |
| TD3 | 0.5 | 1 | independent_policy_evaluations | 33 | 0 | 4096 | 0.441524 | 0.788 |
| TD3 | 0.5 | 2 | training_samples | 4096 | 0 | 3369 | 0.437371 | 0.585 |
| TD3 | 0.5 | 2 | independent_policy_evaluations | 33 | 0 | 4096 | 0.438661 | 0.727 |
| SAC | 1.0 | 0 | training_samples | 4096 | 3 | 3840 | 0.437845 | 0.350 |
| SAC | 1.0 | 0 | independent_policy_evaluations | 33 | 0 | 4096 | 0.44179 | 0.152 |
| SAC | 1.0 | 1 | training_samples | 4096 | 9 | 106 | 0.258809 | 0.346 |
| SAC | 1.0 | 1 | independent_policy_evaluations | 33 | 0 | 3968 | 0.442429 | 0.152 |
| SAC | 1.0 | 2 | training_samples | 4096 | 0 | 4060 | 0.437703 | 0.327 |
| SAC | 1.0 | 2 | independent_policy_evaluations | 33 | 0 | 4096 | 0.442076 | 0.152 |
| SAC | 0.5 | 0 | training_samples | 4096 | 3 | 3840 | 0.438091 | 0.223 |
| SAC | 0.5 | 0 | independent_policy_evaluations | 33 | 0 | 4096 | 0.445025 | 0.000 |
| SAC | 0.5 | 1 | training_samples | 4096 | 9 | 106 | 0.263241 | 0.208 |
| SAC | 0.5 | 1 | independent_policy_evaluations | 33 | 0 | 3968 | 0.44514 | 0.000 |
| SAC | 0.5 | 2 | training_samples | 4096 | 0 | 4060 | 0.437862 | 0.192 |
| SAC | 0.5 | 2 | independent_policy_evaluations | 33 | 0 | 4096 | 0.445069 | 0.000 |
| DE | 1.0 | 0 | actual_candidates | 4096 | 2923 | 3387 | 0.0590214 | 0.240 |
| DE | 1.0 | 1 | actual_candidates | 4096 | 3936 | 3700 | 0.0334161 | 0.002 |
| DE | 1.0 | 2 | actual_candidates | 4096 | 2 | 3964 | 0.437212 | 0.930 |
| DE | 0.5 | 0 | actual_candidates | 4096 | 3948 | 4093 | 0.0228954 | 0.017 |
| DE | 0.5 | 1 | actual_candidates | 4096 | 3966 | 4073 | 0.240807 | 0.001 |
| DE | 0.5 | 2 | actual_candidates | 4096 | 3642 | 4090 | 0.0440016 | 0.077 |
| random_search | 1.0 | 0 | actual_candidates | 4096 | 90 | 603 | 0.149421 | 0.050 |
| random_search | 1.0 | 1 | actual_candidates | 4096 | 98 | 3044 | 0.195769 | 0.053 |
| random_search | 1.0 | 2 | actual_candidates | 4096 | 99 | 1938 | 0.154702 | 0.051 |
| random_search | 0.5 | 0 | actual_candidates | 4096 | 90 | 603 | 0.149421 | 0.050 |
| random_search | 0.5 | 1 | actual_candidates | 4096 | 98 | 483 | 0.234291 | 0.053 |
| random_search | 0.5 | 2 | actual_candidates | 4096 | 99 | 1938 | 0.154702 | 0.051 |

学習サンプル4096点と独立方策評価33点を分けた。全域TD3の発声サンプルは0–1/4096、SACは0–9/4096で、遭遇が極めて少ない。TD3は学習サンプルの約58–63%、SACは約19–35%がf0_lの上限5%領域に入る。これらは行動分布の記述であり、学習アルゴリズム固有の因果の証明ではない。

高f0_lの非持続領域ではE_totalの下側が約0.44に留まる。Oracle断面では発声境界で損失が上昇する一方、非持続領域に入った後のE_specは約0.21で変化が小さく、E_envは上限側へ緩やかに下がる。狭い発声領域への遭遇不足と、この非持続領域内の誤差地形は境界偏りを説明する仮説に整合する。ただし発声だけで低誤差になるわけではなく、critic近似・entropy・探索機構等の寄与を単独に同定していない。報酬の再設計は行わない。

調べた格子では持続702点のE_total中央値0.336499・最小ほぼ0に対し、非持続2686点は最小0.437212・中央値0.516072だった。非持続でE_total<0.3の点は0。従って「既存報酬が発声を全く区別しない」とは結論しない。一方、持続点の最大誤差は0.727039で分布に重なりがあり、発声proxyだけでは正確な模倣を保証しない。Oracleのε_l/ps断面では非持続のf0≈500725でE_total≈0.532778、f0=1e7で≈0.450586となり、非持続領域内でも上限側へ動くと損失が下がる。[条件付き集計と実例](diagnostics/loss_proxy_summary.json)。この有限・緩勾配の領域への方策集中は仮説として整合するが、criticの内部推定が原因だと直接検証したわけではない。

## 3 seed平均±標本SDと発声状態

| 範囲 | 手法 | Final E_total | Selected E_total | Final E_spec | Final E_env | Final発声/3 | Selected発声/3 | 学習発声率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | TD3 | 0.438944 ± 0.00149 | 0.438944 ± 0.00149 | 0.208056 ± 0.000225 | 0.230888 ± 0.00127 | 0 | 0 | 8.13802e-05 ± 0.000141 |
| full | SAC | 0.442125 ± 0.000363 | 0.442098 ± 0.00032 | 0.208146 ± 2.46e-05 | 0.233979 ± 0.000339 | 0 | 0 | 0.000976562 ± 0.00112 |
| full | DE | 0.17655 ± 0.226 | 0.17655 ± 0.226 | 0.0890283 ± 0.103 | 0.0875216 ± 0.124 | 2 | 2 | 0.55835 ± 0.499 |
| full | random_search | 0.502135 ± 0.0382 | 0.166631 ± 0.0254 | 0.211471 ± 0.00227 | 0.290663 ± 0.036 | 0 | 3 | 0.0233561 ± 0.0012 |
| restricted | TD3 | 0.340294 ± 0.157 | 0.338433 ± 0.159 | 0.220936 ± 0.00907 | 0.119358 ± 0.165 | 2 | 2 | 0.534993 ± 0.472 |
| restricted | SAC | 0.25334 ± 0.00692 | 0.244649 ± 0.01 | 0.240647 ± 0.00293 | 0.0126938 ± 0.00399 | 3 | 3 | 0.95166 ± 0.0245 |
| restricted | DE | 0.0216222 ± 0.0187 | 0.0216203 ± 0.0187 | 0.018731 ± 0.0163 | 0.00289119 ± 0.00276 | 3 | 3 | 0.990723 ± 0.00305 |
| restricted | random_search | 0.392561 ± 0.132 | 0.0472222 ± 0.00414 | 0.234569 ± 0.0244 | 0.157992 ± 0.143 | 2 | 3 | 0.316243 ± 0.0086 |

FinalはRL最終決定論的方策、直接法は実4096番目候補。SelectedはRL33独立評価／直接法4096候補からminJを最初の同値で選択した解。λ=1なのでJ=共通E_totalだが、候補集合・評価回数は異なる。直接法の到達性能を見る際はSelectedを確認し、Finalと置換しない。学習中の最良ノイズ付きサンプルを方策の到達性能と偽らない。

| 範囲 | 手法 | seed | Final | Selected | 採用試行 | Final発声 | Selected発声 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | TD3 | 0 | 0.4402334 | 0.4402334 | 4096 | False | False |
| full | TD3 | 1 | 0.4392894 | 0.4392894 | 4096 | False | False |
| full | TD3 | 2 | 0.4373095 | 0.4373095 | 4096 | False | False |
| full | SAC | 0 | 0.44179 | 0.44179 | 4096 | False | False |
| full | SAC | 1 | 0.4425102 | 0.4424289 | 3968 | False | False |
| full | SAC | 2 | 0.4420757 | 0.4420757 | 4096 | False | False |
| full | DE | 0 | 0.05902142 | 0.05902142 | 3387 | True | True |
| full | DE | 1 | 0.03341608 | 0.03341608 | 3700 | True | True |
| full | DE | 2 | 0.4372121 | 0.4372121 | 3964 | False | False |
| full | random_search | 0 | 0.5278518 | 0.1494212 | 603 | False | True |
| full | random_search | 1 | 0.5203306 | 0.1957691 | 3044 | False | True |
| full | random_search | 2 | 0.458222 | 0.1547021 | 1938 | False | True |
| restricted | TD3 | 0 | 0.2334936 | 0.2308455 | 3712 | True | True |
| restricted | TD3 | 1 | 0.5210673 | 0.5210231 | 3968 | False | False |
| restricted | TD3 | 2 | 0.2663199 | 0.2634312 | 3968 | True | True |
| restricted | SAC | 0 | 0.248693 | 0.245489 | 3072 | True | True |
| restricted | SAC | 1 | 0.2612972 | 0.254243 | 2304 | True | True |
| restricted | SAC | 2 | 0.2500312 | 0.2342144 | 768 | True | True |
| restricted | DE | 0 | 3.699679e-05 | 3.699646e-05 | 4095 | True | True |
| restricted | DE | 1 | 0.03172399 | 0.03172399 | 307 | True | True |
| restricted | DE | 2 | 0.03310548 | 0.03309994 | 4089 | True | True |
| restricted | random_search | 0 | 0.382189 | 0.04920684 | 1943 | True | True |
| restricted | random_search | 1 | 0.2657168 | 0.04999159 | 3146 | True | True |
| restricted | random_search | 2 | 0.5297766 | 0.04246813 | 2751 | False | True |

E_spec/E_env、物理量NRMSE、制御値とTargetとの差は[各seed](comparison_per_seed.csv)、[平均SD](comparison_aggregate.csv)、[物理量](parameter_errors.csv)に保存。NRMSEの正規化は両条件とも元の全域物理幅を使用する。n=3の結果から一般的な有意差や優劣を主張しない。

## 計算量と検証

新規学習49,152候補、native98,304,000,000 step。12 CPU workers、並列wall817.42秒。独立訓練時評価210音。RL各4080 gradient iterations、TD3 actor/critic、SAC actor/critic/entropyの更新数を各summaryに分けて保存。

| 範囲 | 手法 | 学習wall秒:平均±SD |
| --- | --- | --- |
| full | TD3 | 875.017 ± 1.23 |
| full | SAC | 880.413 ± 1.41 |
| full | DE | 862.568 ± 1.79 |
| full | random_search | 862.953 ± 0.712 |
| restricted | TD3 | 797.811 ± 0.278 |
| restricted | SAC | 805.21 ± 0.755 |
| restricted | DE | 788.391 ± 1.56 |
| restricted | random_search | 789.317 ± 2.34 |

元全域実験は別の27 worker条件なので、wall差をアルゴリズム固有の速度差としない。物理診断3388音＋代表音7、範囲マッピングテスト4音、追加監査26音は4096予算から別計上。元実験の学習は再実行しない。

監査102,224件を通過。元RLとの初期ネットワーク・16 warmup正規化行動、effective hyperparametersが一致。Randomの4096正規化候補列、DEの初期8正規化候補も対照と同一。物理f0_lの写像だけが異なる。新モデルは別プロセスで12回再読み込みし、Final/Best計24音の行動・制御・波形・評価差0を確認。loss/gradient/parameterのNaN/Infで停止する既存検査を維持し、全runが完了した。

## 図の読み方

各誤差図は同一範囲・λ=1の4手法を重ねる。全域と制限域は別画像で共通軸。実線4pt、SciencePlots、軸46pt・目盛30pt・凡例29pt。RL33点と直接法4096点を全て描き、平滑化・頂点省略なし。帯は標本SD。
音響図はmagma、Target最大PSDの共通基準、範囲(-70.0, 10.0) dB、共通カラーバー。5 ms RMSは双方を同じTargetピークで割り、縦軸(0.0, 0.6080270932969052)。表示は10 MHzへupsampleしたdensity PSDで、学習の元50 kHz・paddingありspectrum PSDとは別。音圧校正dB SPLではなく、upsampleで情報は増えない。
表示PSDは従来同様にフレーム平均を除去するため、DC/緩い過渡成分は暗くてもRMSに残り得る。RMSとAC発声proxyを区別し、代表波形の全区間と終盤20msを共通振幅で併記した。
パラメータ図はFinal/Selected別、行が係数・列が全域/制限域。λは両列とも1であり、重み比較を装っていない。各seed・平均±標本SD・Target破線を表示する。[図の全索引](FIGURES.md)。

## 今回の結果から言える範囲

SACのFinal発声は0/3→3/3、TD3は0/3→2/3になった。Random Searchの経験的発声率は約2.34%→31.6%で、制限範囲が発声点への遭遇頻度を増やしたことを支持する。この一様乱数探索の実サンプル率と、非一様診断格子の702/3388は区別する。

SACの平均Final E_envは約0.234→0.0127まで低下した一方、E_specは約0.208→0.241へ増加した。発声・振幅包絡の改善と、スペクトルの正確な模倣は同じではない。TD3 seed1は制限後も非持続で、Final E_totalは約0.439→0.521へ悪化している。平均改善だけで全seedの解決としない。

DE/RandomのSelected E_totalも約0.1766→0.02162 / 0.1666→0.04722へ低下した。改善はRL固有ではなく、同じ領域制約が直接法にも有効だった。制限後もDEが最も低い平均誤差で、DEのうち2 seedには約0.03の局所解が残る。制御値の同定と音響誤差の低さも別に判断する。

次の切り分けでは、まず保存済みcriticの予測と保存実測Jの対応を再学習なしで調べることを候補とする。その後、同じ制限範囲・目標・λ=1・seed・4096・ネットワーク・学習率を固定し、探索幅など1要因だけを変える追試を計画する。今回はその追加学習や報酬変更を行っていない。

## 判断と次の課題

制限後の改善は探索制約の効果として扱い、元の広域条件からRL自体の優位を結論しない。改善/非改善・非持続seedも全て表へ含めた。制限しても誤差が残る場合、次はまず探索機構または物理パラメータの尺度を1要因ずつ切り分ける計画を立てる。発声proxyを報酬へ追加する案は別実験であり、この診断で無断導入しない。未知目標・時変制御・複数音節・方策再利用を公平に比較するための準備とし、そこでRLがDEに勝つという結論は先取りしない。

## 再実行

リポジトリ直下、既存`.venv-rl`を使用。新規同名結果は上書き拒否する。
```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-newissues-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue18_diagnostics survey --workers 9
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue18_diagnostics reanalyze
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue18_training prepare
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue18_training train --workers 12
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue18_audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue18_report
```
既存結果がある作業コピーではaudit/reportのみ再実行できる。学習再現は空の同名出力を使う別作業コピーで行い、既存正本を削除しない。commit/pushは行っていない。


## 実行セッション中断と再実行の計上

最初の12 runは実行セッションがSIGTERM（終了143）で終了し、全runの最後の進捗記録は3200候補だった。終了の外的な正確な原因は未確認。損失・NaN/Inf例外として扱わず、完了4096の比較へ混ぜない。少なくとも38,400学習候補と156方策評価が実行され、最後の記録以後の厳密候補数は復元できない。

中断checkpointには完全なreplay buffer/RNG状態がないため、同じseed・条件で12 run全てを最初から実行した。中断時の全記録は [interrupted_attempt_1](interrupted_attempt_1/interruption.json) に保持し、再開を装わない。再実行の48 checkpointで行動・物理値・誤差が中断試行と完全一致することも監査した。本文の49,152候補・wall時間は完了した比較用試行のみで、中断分の計算費用を除外して速さを主張しない。[全計上](total_accounting.json)。
