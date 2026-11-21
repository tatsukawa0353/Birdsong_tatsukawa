# Issue21 — Case A・一定3Dのネットワーク幅比較

[結果レポート](REPORT.md)。λ=1、Actor/Critic幅64/128/256の18run。32幅はIssue18をreadonly参照。

- [条件・source hash](manifest.json)、[事前検証](preflight.json)
- [全seed](comparison_per_seed.csv)、[平均±標本SD](comparison_aggregate.csv)、[対応seed差](paired_width_deltas.csv)
- [独立方策評価](policy_evaluations.csv)、[新規学習サンプル](new_training_samples.csv.gz)
- [Critic指標](critic_metrics.csv)、[Actor比較](actor_diagnostics.csv)、[推論記録](critic_execution.json)
- [再生監査](audit.json)、[旧成果の最終照合](integrity.json)、[ファイルhash](artifact_checksums.json)
- [図8枚の出所](figure_sources.json)、[元波形/モデル参照](endpoint_sources.json)
- [聴取用WAV ZIP](new_listening_audio.zip)、[gainと対応表](listening_audio.json)
- [再現と監査](REPRODUCE.md)、[変更前](storage_before.json)、[変更後](storage_after.json)
