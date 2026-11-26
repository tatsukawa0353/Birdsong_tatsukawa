# Issue23 — SAC128・λ=.75の安定性診断

> 2026-11-26 成果物整理：本文の計算結果・実施/未実施は当時の記録です。現在の配置・削除した再生成図・hash検証は[移行記録](../../../maintenance/20261126/README.md)を参照。旧manifest/hashは原本を維持しています。


[REPORT](REPORT.md) / [学習前のStage A](STAGE_A.md) / [再現手順](REPRODUCE.md) / [方策索引](checkpoint_index.json)

旧Issue22の3seedを設定変更なしで再現し、中間Critic・Actor更新を診断した。Final、minJのSelected、minE_totalのposthocを別々に保持する。保存による利得と学習改善を区別する。過去成果物は読み取り専用。
