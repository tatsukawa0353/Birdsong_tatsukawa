# Phase1 — 4手法×2λの比較

最初に [REPORT.md](REPORT.md)。次に下表の6条件を確認する。各画像は同じ次元・同じλの4手法を表示する。
旧単独手法の誤差図は残していない。次元間・λ間を同じ画像には混在させない。

| 次元 | λ | 実評価:seed0 | 実評価:平均±SD | J採用解:seed0 | J採用解:平均±SD |
| --- | --- | --- | --- | --- | --- |
| 1d | 1.0 | [seed0](figures/learning_curves/1d_lambda1_seed0.png) | [平均±SD](figures/learning_curves/1d_lambda1_mean_sd.png) | [seed0](figures/best_observed/1d_lambda1_seed0.png) | [平均±SD](figures/best_observed/1d_lambda1_mean_sd.png) |
| 1d | 0.5 | [seed0](figures/learning_curves/1d_lambda0p5_seed0.png) | [平均±SD](figures/learning_curves/1d_lambda0p5_mean_sd.png) | [seed0](figures/best_observed/1d_lambda0p5_seed0.png) | [平均±SD](figures/best_observed/1d_lambda0p5_mean_sd.png) |
| 2d | 1.0 | [seed0](figures/learning_curves/2d_lambda1_seed0.png) | [平均±SD](figures/learning_curves/2d_lambda1_mean_sd.png) | [seed0](figures/best_observed/2d_lambda1_seed0.png) | [平均±SD](figures/best_observed/2d_lambda1_mean_sd.png) |
| 2d | 0.5 | [seed0](figures/learning_curves/2d_lambda0p5_seed0.png) | [平均±SD](figures/learning_curves/2d_lambda0p5_mean_sd.png) | [seed0](figures/best_observed/2d_lambda0p5_seed0.png) | [平均±SD](figures/best_observed/2d_lambda0p5_mean_sd.png) |
| 3d | 1.0 | [seed0](figures/learning_curves/3d_lambda1_seed0.png) | [平均±SD](figures/learning_curves/3d_lambda1_mean_sd.png) | [seed0](figures/best_observed/3d_lambda1_seed0.png) | [平均±SD](figures/best_observed/3d_lambda1_mean_sd.png) |
| 3d | 0.5 | [seed0](figures/learning_curves/3d_lambda0p5_seed0.png) | [平均±SD](figures/learning_curves/3d_lambda0p5_mean_sd.png) | [seed0](figures/best_observed/3d_lambda0p5_seed0.png) | [平均±SD](figures/best_observed/3d_lambda0p5_mean_sd.png) |

実評価はRLの現方策と直接法の実候補。J採用解図はminJλで選ぶ解の共通E_totalであり、事後minE_totalではない。
RLは33回の独立評価、DE/Randomは4096候補。候補数・評価数の違いはREPORTを参照。

## 数表・条件・検証

- [最終/選択解の平均±標本SD](comparison_aggregate.csv) / [各seed](comparison_per_seed.csv)
- [全曲線](comparison_curves.csv) / [指定候補数](checkpoint_comparison.csv) / [λ差](lambda_effects.csv)
- [実物理量と目標との差](parameter_errors.csv)
- [追加条件・データ再利用](direct_weight_extension/manifest.json) / [追加実行時間](direct_weight_extension/execution.json)
- [現在の追加監査](direct_weight_extension/audit.json) / [元54runの監査](audit.json) / [図の検査](figure_audit.json)
- [削除した旧単独手法図72枚](direct_weight_extension/deleted_single_curve_images.txt)
- [元54runの設定](config.json) / [元manifest](manifest.json) / [追加設定](direct_weight_extension/config.json)

## 音響図・物理量図（seed0、実候補4096／最終方策）

スペクトログラムは上段Target・下段Generated。既存図は変更せず保持し、新しいDE/Random λ=.5を追加した。

| 次元 | 手法 | λ | スペクトログラム | 物理制御 |
| --- | --- | --- | --- | --- |
| 1d | TD3 | 1.0 | [図](figures/spectrograms/1d_TD3_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/1d_TD3_lambda1_epsilon_l_seed0.png) |
| 1d | SAC | 1.0 | [図](figures/spectrograms/1d_SAC_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/1d_SAC_lambda1_epsilon_l_seed0.png) |
| 1d | DE | 1.0 | [図](figures/spectrograms/1d_DE_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/1d_DE_lambda1_epsilon_l_seed0.png) |
| 1d | random_search | 1.0 | [図](figures/spectrograms/1d_random_search_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/1d_random_search_lambda1_epsilon_l_seed0.png) |
| 1d | TD3 | 0.5 | [図](figures/spectrograms/1d_TD3_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/1d_TD3_lambda0p5_epsilon_l_seed0.png) |
| 1d | SAC | 0.5 | [図](figures/spectrograms/1d_SAC_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/1d_SAC_lambda0p5_epsilon_l_seed0.png) |
| 1d | DE | 0.5 | [図](figures/spectrograms/1d_DE_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/1d_DE_lambda0p5_epsilon_l_seed0.png) |
| 1d | random_search | 0.5 | [図](figures/spectrograms/1d_random_search_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/1d_random_search_lambda0p5_epsilon_l_seed0.png) |
| 2d | TD3 | 1.0 | [図](figures/spectrograms/2d_TD3_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/2d_TD3_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/2d_TD3_lambda1_ps_seed0.png) |
| 2d | SAC | 1.0 | [図](figures/spectrograms/2d_SAC_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/2d_SAC_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/2d_SAC_lambda1_ps_seed0.png) |
| 2d | DE | 1.0 | [図](figures/spectrograms/2d_DE_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/2d_DE_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/2d_DE_lambda1_ps_seed0.png) |
| 2d | random_search | 1.0 | [図](figures/spectrograms/2d_random_search_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/2d_random_search_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/2d_random_search_lambda1_ps_seed0.png) |
| 2d | TD3 | 0.5 | [図](figures/spectrograms/2d_TD3_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/2d_TD3_lambda0p5_epsilon_l_seed0.png) / [ps](figures/physical_controls/2d_TD3_lambda0p5_ps_seed0.png) |
| 2d | SAC | 0.5 | [図](figures/spectrograms/2d_SAC_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/2d_SAC_lambda0p5_epsilon_l_seed0.png) / [ps](figures/physical_controls/2d_SAC_lambda0p5_ps_seed0.png) |
| 2d | DE | 0.5 | [図](figures/spectrograms/2d_DE_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/2d_DE_lambda0p5_epsilon_l_seed0.png) / [ps](figures/physical_controls/2d_DE_lambda0p5_ps_seed0.png) |
| 2d | random_search | 0.5 | [図](figures/spectrograms/2d_random_search_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/2d_random_search_lambda0p5_epsilon_l_seed0.png) / [ps](figures/physical_controls/2d_random_search_lambda0p5_ps_seed0.png) |
| 3d | TD3 | 1.0 | [図](figures/spectrograms/3d_TD3_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/3d_TD3_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/3d_TD3_lambda1_ps_seed0.png) / [f0_l](figures/physical_controls/3d_TD3_lambda1_f0_l_seed0.png) |
| 3d | SAC | 1.0 | [図](figures/spectrograms/3d_SAC_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/3d_SAC_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/3d_SAC_lambda1_ps_seed0.png) / [f0_l](figures/physical_controls/3d_SAC_lambda1_f0_l_seed0.png) |
| 3d | DE | 1.0 | [図](figures/spectrograms/3d_DE_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/3d_DE_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/3d_DE_lambda1_ps_seed0.png) / [f0_l](figures/physical_controls/3d_DE_lambda1_f0_l_seed0.png) |
| 3d | random_search | 1.0 | [図](figures/spectrograms/3d_random_search_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/3d_random_search_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/3d_random_search_lambda1_ps_seed0.png) / [f0_l](figures/physical_controls/3d_random_search_lambda1_f0_l_seed0.png) |
| 3d | TD3 | 0.5 | [図](figures/spectrograms/3d_TD3_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/3d_TD3_lambda0p5_epsilon_l_seed0.png) / [ps](figures/physical_controls/3d_TD3_lambda0p5_ps_seed0.png) / [f0_l](figures/physical_controls/3d_TD3_lambda0p5_f0_l_seed0.png) |
| 3d | SAC | 0.5 | [図](figures/spectrograms/3d_SAC_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/3d_SAC_lambda0p5_epsilon_l_seed0.png) / [ps](figures/physical_controls/3d_SAC_lambda0p5_ps_seed0.png) / [f0_l](figures/physical_controls/3d_SAC_lambda0p5_f0_l_seed0.png) |
| 3d | DE | 0.5 | [図](figures/spectrograms/3d_DE_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/3d_DE_lambda0p5_epsilon_l_seed0.png) / [ps](figures/physical_controls/3d_DE_lambda0p5_ps_seed0.png) / [f0_l](figures/physical_controls/3d_DE_lambda0p5_f0_l_seed0.png) |
| 3d | random_search | 0.5 | [図](figures/spectrograms/3d_random_search_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/3d_random_search_lambda0p5_epsilon_l_seed0.png) / [ps](figures/physical_controls/3d_random_search_lambda0p5_ps_seed0.png) / [f0_l](figures/physical_controls/3d_random_search_lambda0p5_f0_l_seed0.png) |

モデル・音声・rawは runs/<次元>/<手法>/<lambda>/seed_<n>/。新直接法のbest_objectiveとbest_common_posthocを区別する。
Random λ=.5のsource_reuse.jsonに候補プール・音声の共有元と新規選択音を記録した。
