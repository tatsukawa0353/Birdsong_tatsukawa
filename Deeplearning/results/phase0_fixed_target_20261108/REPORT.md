# Phase 0：固定目標と報酬アブレーション

正本ブランチはmain_deeplearning。保存名20261108は歴史的な名前で、実行日は各manifestを参照。

- [Issue #15の比較レポート](reward_ablation/REPORT.md)：A/B/C、3手法×3 seed、共通評価。
- [Issue #14のA長期学習レポート（保存時の原文）](CONTROL_REPORT.md)
- [再学習なしの成分曲線](reward_ablation/reanalysis/components_mean_sd.png)
- [A/B/Cの学習曲線](reward_ablation/figures/learning_curves_seed0.png)
- [旧ランダム目標の実データ](legacy_random_target/)：移設前と全bytes一致。

| 条件 | 手法 | 最終共通誤差 平均±標本SD | 周波数MAE[Hz] | 振幅MAE |
| --- | --- | --- | --- | --- |
| combined | PPO | 0.030880 ± 0.006337 | 95.655 | 0.007045 |
| combined | TD3 | 0.020582 ± 0.001583 | 30.511 | 0.001885 |
| combined | SAC | 0.029000 ± 0.004149 | 68.668 | 0.001744 |
| spectral_only | PPO | 0.061926 ± 0.006444 | 83.118 | 0.045653 |
| spectral_only | TD3 | 0.056940 ± 0.005956 | 22.030 | 0.033296 |
| spectral_only | SAC | 0.070406 ± 0.010367 | 16.956 | 0.046826 |
| envelope_only | PPO | 0.064683 ± 0.019779 | 481.630 | 0.005762 |
| envelope_only | TD3 | 0.084909 ± 0.011404 | 703.458 | 0.001849 |
| envelope_only | SAC | 0.058296 ± 0.000578 | 367.755 | 0.000710 |

Aは保存済み9 runを再利用。B/Cはそれぞれ9 runをゼロから102,400環境step、2,048step毎に評価。報酬項の選択以外は共通。主評価は最終方策の全0.35秒距離とf/A MAEで、局所報酬や途中最良とは区別した。

スカラーの数値スケールが変わるため、差を項の干渉だけで説明しない。λ調整・カリキュラムは未検証で、追加学習は行っていない。固定正弦波のin-target成績であり、未知音・鳴管・実録音への一般化は主張しない。

親直下のAモデル・音声・曲線・JSONと既存図は保持した。過去のmanifest/CSV/復元記録にある旧results/phase0パスは履歴で、現在はlegacy_random_targetへ対応する。監査・復元用resolverは旧パスのhash値を変更せず解決する。
