# Issue27 — Case B自己Target：preflight停止

> 2026-11-26 成果物整理：本文の計算結果・実施/未実施は当時の記録です。現在の配置・削除した再生成図・hash検証は[移行記録](../../../../maintenance/20261126/README.md)を参照。旧manifest/hashは原本を維持しています。


指定Targetは持続発声proxy0.0003395（基準>.05）で、Issueの指示に従いSACは0run。左右α/β非ゼロ・右f0=1e7・oracle一致・右音響寄与は確認済み。

[診断と次の判断](REPORT.md) / [事前計画](PLAN.md) / [再現](REPRODUCE.md) / [manifest](manifest.json)

[Target B WAV](audio/oracle.wav) / [B oracle WAV](audio/oracle.wav) / [同一制御A対照 WAV](audio/same_controls_caseA.wav) / [音響指標](sound_metrics.csv) / [左右末尾状態](tail_motion.csv) / [source項](native_block_statistics.csv)

[新規B生波形・制御](preflight/target.npz) / [native詳細B](preflight/oracle.npz) / [A対照詳細](preflight/same_controls_caseA.npz) / [検証](diagnostic_audit.json) / [未開始会計](training_accounting.csv) / [保存検査](integrity.json)
