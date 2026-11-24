# Issue25 — Critic LR：オンラインと固定データを分離

[REPORT](REPORT.md) / [計算前PLAN](PLAN.md) / [再現手順](REPRODUCE.md) / [並列の評価軸](parallel_online_offline.csv) / [WAV ZIP](listening.zip)

Actor/温度LRを固定したSAC6runと、既存データ固定のCritic-only9本。Final音響精度とCritic予測精度を混同しない。基準と旧成果物は読み取り専用。
