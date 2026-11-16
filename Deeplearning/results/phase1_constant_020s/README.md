# Phase1 — 4手法×2λの比較

最初に [REPORT.md](REPORT.md)。次に下表の6条件を確認する。各画像は同じ次元・同じλの4手法を表示する。
旧単独手法の誤差図は残していない。次元間・λ間を同じ画像には混在させない。

誤差線図は4手法すべて実線・線幅4.0pt。

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

## 一定制御パラメータの集約比較

| 次元 | Final | Selected Best |
| --- | --- | --- |
| 1d | [集約図](figures/physical_controls/1d_final.png) | [集約図](figures/physical_controls/1d_selected_best.png) |
| 2d | [集約図](figures/physical_controls/2d_final.png) | [集約図](figures/physical_controls/2d_selected_best.png) |
| 3d | [集約図](figures/physical_controls/3d_final.png) | [集約図](figures/physical_controls/3d_selected_best.png) |

各図はλの2列、可変係数の行。seed0/1/2と平均±標本SD、目標の破線を表示。f0_lはHzではない。

## Final / Selected BestのスペクトログラムとRMS包絡

[全次元・手法・λ・seedの図一覧](visualization_revision/FIGURES.md)。Final PSD、Selected PSD、Final RMS、Selected RMSを別リンクで案内する。

全図でTarget最大PSDを0 dBとする共通基準・共通範囲・magmaを使用。RMSは双方を同じTargetピークで正規化。校正済みdB SPLではない。

[描画条件](visualization_revision/display_config.json) / [出所一覧](visualization_revision/catalog.json) / [監査](visualization_revision/audit.json) / [削除一覧](visualization_revision/deleted_images.txt)。
