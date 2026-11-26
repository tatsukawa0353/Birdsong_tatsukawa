# Issue25 — Critic LR：オンラインと固定データを分離

> 2026-11-26 成果物整理：本文の計算結果・実施/未実施は当時の記録です。現在の配置・削除した再生成図・hash検証は[移行記録](../../../../maintenance/20261126/README.md)を参照。旧manifest/hashは原本を維持しています。


[REPORT](REPORT.md) / [計算前PLAN](PLAN.md) / [再現手順](REPRODUCE.md) / [並列の評価軸](parallel_online_offline.csv) / [WAV ZIP](listening.zip)

Actor/温度LRを固定したSAC6runと、既存データ固定のCritic-only9本。Final音響精度とCritic予測精度を混同しない。基準と旧成果物は読み取り専用。
