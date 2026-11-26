# Issue24 — SAC128・λ=.75のActor学習率比較

> 2026-11-26 成果物整理：本文の計算結果・実施/未実施は当時の記録です。現在の配置・削除した再生成図・hash検証は[移行記録](../../../../maintenance/20261126/README.md)を参照。旧manifest/hashは原本を維持しています。


[REPORT](REPORT.md) / [再現とオフライン確認](REPRODUCE.md) / [全endpoint](comparison_per_seed.csv) / [到達・維持](stability_and_acquisition.csv) / [WAV ZIP](listening.zip)

既存Actor3e-4を読み取り専用で参照し、Actor1e-4/3e-5を各3seed×4096候補比較する。Critic/温度のLRは3e-4。Finalを主評価とし、minJ/minE保存・学習経験・独立方策の到達・維持を区別する。
