# Issue22 — SAC128の報酬係数和1比較

> 2026-11-26 成果物整理：本文の計算結果・実施/未実施は当時の記録です。現在の配置・削除した再生成図・hash検証は[移行記録](../../../../maintenance/20261126/README.md)を参照。旧manifest/hashは原本を維持しています。


[結果と採用判断](REPORT.md)。λ=.50/.75/.90の9run、旧Issue21の1+1報酬はreadonly参考。

- [条件・source hash](manifest.json)、[係数和と物理の事前検証](preflight.json)
- [全seed](comparison_per_seed.csv)、[平均±標本SD](comparison_aggregate.csv)、[対応seed差](paired_differences.csv)
- [Selected−Final](selected_final_stability.csv)、[33点の方策評価](policy_evaluations.csv)
- [α・分散・entropy](policy_dynamics.csv)、[学習サンプル](training_samples.csv.gz)
- [補助Critic指標](critic_metrics.csv)、[診断条件](critic_execution.json)
- [モデル/行動再生監査](audit.json)、[旧成果・リンク・CSV整合](integrity.json)、[成果物hash](artifact_checksums.json)
- [図8枚の出所](figure_sources.json)、[モデル/波形の出所](endpoint_sources.json)
- [聴取用WAV ZIP](new_listening_audio.zip)、[共通gainと対応表](listening_audio.json)
- [再現手順](REPRODUCE.md)、[変更前](storage_before.json)、[変更後](storage_after.json)
