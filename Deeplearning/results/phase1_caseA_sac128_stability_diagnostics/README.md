# Issue23 — SAC128・λ=.75の安定性診断

[REPORT](REPORT.md) / [学習前のStage A](STAGE_A.md) / [再現手順](REPRODUCE.md) / [方策索引](checkpoint_index.json)

旧Issue22の3seedを設定変更なしで再現し、中間Critic・Actor更新を診断した。Final、minJのSelected、minE_totalのposthocを別々に保持する。保存による利得と学習改善を区別する。過去成果物は読み取り専用。
