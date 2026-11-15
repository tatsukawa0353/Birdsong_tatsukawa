# Phase 1 — 4096回の正本

最初に [REPORT.md](REPORT.md) を読み、次に以下の曲線→スペクトログラム→物理制御→CSVの順で確認する。
各画像は1条件だけ。seed0と平均±標本SDは別ファイル。DE/Randomはλ=1のみ。

## 各次元・手法・λの単一曲線

| 次元 | 手法 | λ | seed0 | 平均±SD | best seed0 | best 平均±SD |
| --- | --- | --- | --- | --- | --- | --- |
| 1d | DE | 1.0 | [図](figures/learning_curves/1d_DE_lambda1_seed0.png) | [図](figures/learning_curves/1d_DE_lambda1_mean_sd.png) | [図](figures/best_observed/1d_DE_lambda1_seed0.png) | [図](figures/best_observed/1d_DE_lambda1_mean_sd.png) |
| 1d | SAC | 0.5 | [図](figures/learning_curves/1d_SAC_lambda0p5_seed0.png) | [図](figures/learning_curves/1d_SAC_lambda0p5_mean_sd.png) | [図](figures/best_observed/1d_SAC_lambda0p5_seed0.png) | [図](figures/best_observed/1d_SAC_lambda0p5_mean_sd.png) |
| 1d | SAC | 1.0 | [図](figures/learning_curves/1d_SAC_lambda1_seed0.png) | [図](figures/learning_curves/1d_SAC_lambda1_mean_sd.png) | [図](figures/best_observed/1d_SAC_lambda1_seed0.png) | [図](figures/best_observed/1d_SAC_lambda1_mean_sd.png) |
| 1d | TD3 | 0.5 | [図](figures/learning_curves/1d_TD3_lambda0p5_seed0.png) | [図](figures/learning_curves/1d_TD3_lambda0p5_mean_sd.png) | [図](figures/best_observed/1d_TD3_lambda0p5_seed0.png) | [図](figures/best_observed/1d_TD3_lambda0p5_mean_sd.png) |
| 1d | TD3 | 1.0 | [図](figures/learning_curves/1d_TD3_lambda1_seed0.png) | [図](figures/learning_curves/1d_TD3_lambda1_mean_sd.png) | [図](figures/best_observed/1d_TD3_lambda1_seed0.png) | [図](figures/best_observed/1d_TD3_lambda1_mean_sd.png) |
| 1d | Random | 1.0 | [図](figures/learning_curves/1d_random_search_lambda1_seed0.png) | [図](figures/learning_curves/1d_random_search_lambda1_mean_sd.png) | [図](figures/best_observed/1d_random_search_lambda1_seed0.png) | [図](figures/best_observed/1d_random_search_lambda1_mean_sd.png) |
| 2d | DE | 1.0 | [図](figures/learning_curves/2d_DE_lambda1_seed0.png) | [図](figures/learning_curves/2d_DE_lambda1_mean_sd.png) | [図](figures/best_observed/2d_DE_lambda1_seed0.png) | [図](figures/best_observed/2d_DE_lambda1_mean_sd.png) |
| 2d | SAC | 0.5 | [図](figures/learning_curves/2d_SAC_lambda0p5_seed0.png) | [図](figures/learning_curves/2d_SAC_lambda0p5_mean_sd.png) | [図](figures/best_observed/2d_SAC_lambda0p5_seed0.png) | [図](figures/best_observed/2d_SAC_lambda0p5_mean_sd.png) |
| 2d | SAC | 1.0 | [図](figures/learning_curves/2d_SAC_lambda1_seed0.png) | [図](figures/learning_curves/2d_SAC_lambda1_mean_sd.png) | [図](figures/best_observed/2d_SAC_lambda1_seed0.png) | [図](figures/best_observed/2d_SAC_lambda1_mean_sd.png) |
| 2d | TD3 | 0.5 | [図](figures/learning_curves/2d_TD3_lambda0p5_seed0.png) | [図](figures/learning_curves/2d_TD3_lambda0p5_mean_sd.png) | [図](figures/best_observed/2d_TD3_lambda0p5_seed0.png) | [図](figures/best_observed/2d_TD3_lambda0p5_mean_sd.png) |
| 2d | TD3 | 1.0 | [図](figures/learning_curves/2d_TD3_lambda1_seed0.png) | [図](figures/learning_curves/2d_TD3_lambda1_mean_sd.png) | [図](figures/best_observed/2d_TD3_lambda1_seed0.png) | [図](figures/best_observed/2d_TD3_lambda1_mean_sd.png) |
| 2d | Random | 1.0 | [図](figures/learning_curves/2d_random_search_lambda1_seed0.png) | [図](figures/learning_curves/2d_random_search_lambda1_mean_sd.png) | [図](figures/best_observed/2d_random_search_lambda1_seed0.png) | [図](figures/best_observed/2d_random_search_lambda1_mean_sd.png) |
| 3d | DE | 1.0 | [図](figures/learning_curves/3d_DE_lambda1_seed0.png) | [図](figures/learning_curves/3d_DE_lambda1_mean_sd.png) | [図](figures/best_observed/3d_DE_lambda1_seed0.png) | [図](figures/best_observed/3d_DE_lambda1_mean_sd.png) |
| 3d | SAC | 0.5 | [図](figures/learning_curves/3d_SAC_lambda0p5_seed0.png) | [図](figures/learning_curves/3d_SAC_lambda0p5_mean_sd.png) | [図](figures/best_observed/3d_SAC_lambda0p5_seed0.png) | [図](figures/best_observed/3d_SAC_lambda0p5_mean_sd.png) |
| 3d | SAC | 1.0 | [図](figures/learning_curves/3d_SAC_lambda1_seed0.png) | [図](figures/learning_curves/3d_SAC_lambda1_mean_sd.png) | [図](figures/best_observed/3d_SAC_lambda1_seed0.png) | [図](figures/best_observed/3d_SAC_lambda1_mean_sd.png) |
| 3d | TD3 | 0.5 | [図](figures/learning_curves/3d_TD3_lambda0p5_seed0.png) | [図](figures/learning_curves/3d_TD3_lambda0p5_mean_sd.png) | [図](figures/best_observed/3d_TD3_lambda0p5_seed0.png) | [図](figures/best_observed/3d_TD3_lambda0p5_mean_sd.png) |
| 3d | TD3 | 1.0 | [図](figures/learning_curves/3d_TD3_lambda1_seed0.png) | [図](figures/learning_curves/3d_TD3_lambda1_mean_sd.png) | [図](figures/best_observed/3d_TD3_lambda1_seed0.png) | [図](figures/best_observed/3d_TD3_lambda1_mean_sd.png) |
| 3d | Random | 1.0 | [図](figures/learning_curves/3d_random_search_lambda1_seed0.png) | [図](figures/learning_curves/3d_random_search_lambda1_mean_sd.png) | [図](figures/best_observed/3d_random_search_lambda1_seed0.png) | [図](figures/best_observed/3d_random_search_lambda1_mean_sd.png) |

## 目標/生成スペクトログラムと物理制御

スペクトログラムは上段Target・下段Generated。制御図は実際の選択値1本で、目標値はREPORTの条件表を参照する。

| 次元 | 手法 | λ | スペクトログラム | 選択物理量 |
| --- | --- | --- | --- | --- |
| 1d | DE | 1.0 | [図](figures/spectrograms/1d_DE_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/1d_DE_lambda1_epsilon_l_seed0.png) |
| 1d | SAC | 0.5 | [図](figures/spectrograms/1d_SAC_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/1d_SAC_lambda0p5_epsilon_l_seed0.png) |
| 1d | SAC | 1.0 | [図](figures/spectrograms/1d_SAC_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/1d_SAC_lambda1_epsilon_l_seed0.png) |
| 1d | TD3 | 0.5 | [図](figures/spectrograms/1d_TD3_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/1d_TD3_lambda0p5_epsilon_l_seed0.png) |
| 1d | TD3 | 1.0 | [図](figures/spectrograms/1d_TD3_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/1d_TD3_lambda1_epsilon_l_seed0.png) |
| 1d | Random | 1.0 | [図](figures/spectrograms/1d_random_search_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/1d_random_search_lambda1_epsilon_l_seed0.png) |
| 2d | DE | 1.0 | [図](figures/spectrograms/2d_DE_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/2d_DE_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/2d_DE_lambda1_ps_seed0.png) |
| 2d | SAC | 0.5 | [図](figures/spectrograms/2d_SAC_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/2d_SAC_lambda0p5_epsilon_l_seed0.png) / [ps](figures/physical_controls/2d_SAC_lambda0p5_ps_seed0.png) |
| 2d | SAC | 1.0 | [図](figures/spectrograms/2d_SAC_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/2d_SAC_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/2d_SAC_lambda1_ps_seed0.png) |
| 2d | TD3 | 0.5 | [図](figures/spectrograms/2d_TD3_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/2d_TD3_lambda0p5_epsilon_l_seed0.png) / [ps](figures/physical_controls/2d_TD3_lambda0p5_ps_seed0.png) |
| 2d | TD3 | 1.0 | [図](figures/spectrograms/2d_TD3_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/2d_TD3_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/2d_TD3_lambda1_ps_seed0.png) |
| 2d | Random | 1.0 | [図](figures/spectrograms/2d_random_search_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/2d_random_search_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/2d_random_search_lambda1_ps_seed0.png) |
| 3d | DE | 1.0 | [図](figures/spectrograms/3d_DE_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/3d_DE_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/3d_DE_lambda1_ps_seed0.png) / [f0_l](figures/physical_controls/3d_DE_lambda1_f0_l_seed0.png) |
| 3d | SAC | 0.5 | [図](figures/spectrograms/3d_SAC_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/3d_SAC_lambda0p5_epsilon_l_seed0.png) / [ps](figures/physical_controls/3d_SAC_lambda0p5_ps_seed0.png) / [f0_l](figures/physical_controls/3d_SAC_lambda0p5_f0_l_seed0.png) |
| 3d | SAC | 1.0 | [図](figures/spectrograms/3d_SAC_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/3d_SAC_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/3d_SAC_lambda1_ps_seed0.png) / [f0_l](figures/physical_controls/3d_SAC_lambda1_f0_l_seed0.png) |
| 3d | TD3 | 0.5 | [図](figures/spectrograms/3d_TD3_lambda0p5_seed0.png) | [epsilon_l](figures/physical_controls/3d_TD3_lambda0p5_epsilon_l_seed0.png) / [ps](figures/physical_controls/3d_TD3_lambda0p5_ps_seed0.png) / [f0_l](figures/physical_controls/3d_TD3_lambda0p5_f0_l_seed0.png) |
| 3d | TD3 | 1.0 | [図](figures/spectrograms/3d_TD3_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/3d_TD3_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/3d_TD3_lambda1_ps_seed0.png) / [f0_l](figures/physical_controls/3d_TD3_lambda1_f0_l_seed0.png) |
| 3d | Random | 1.0 | [図](figures/spectrograms/3d_random_search_lambda1_seed0.png) | [epsilon_l](figures/physical_controls/3d_random_search_lambda1_epsilon_l_seed0.png) / [ps](figures/physical_controls/3d_random_search_lambda1_ps_seed0.png) / [f0_l](figures/physical_controls/3d_random_search_lambda1_f0_l_seed0.png) |

## 数表・検証・データ

- [最終値の平均・標本SD](comparison_aggregate.csv) / [個別seed](comparison_per_seed.csv)
- [全曲線の生データ](comparison_curves.csv) / [指定checkpoint](checkpoint_comparison.csv) / [λの対応差](lambda_effects.csv)
- [target診断](diagnostics/review.json) / [設定](config.json) / [manifest](manifest.json)
- [独立監査](audit.json) / [図の仕様監査](figure_audit.json) / [図と数値の対応](figure_sources.json)
- [実行時間](execution.json) / [描画時間](report_timing.json) / [整理一覧](CLEANUP.md) / [歴史資料](history/README.md)

raw音声・モデルは runs/<次元>/<手法>/<lambda1またはlambda0p5>/seed_<n>/。
initial/、best/、最終（run直下）、checkpoints/<回数>/を区別する。
旧256回の詳細ログ・重複図は新しい正本に混在させていない。

同じ数値条件を再実行するコマンドはREPORTの「実行・再検証」。issue17_replayは新IDを必須とし、結果を正本内reproductions/へ保存する。
