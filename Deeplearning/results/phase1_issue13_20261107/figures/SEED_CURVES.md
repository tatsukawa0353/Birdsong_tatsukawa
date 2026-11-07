# seed別の誤差線図

各図は1つのseedの4手法。全実線・線幅2.8・縦軸上限0.6・同じ表示範囲。実測値を変更せず、線端と接合部を丸めて描画した。平均化・数値平滑化・補間は行っていない。

| seed | 一定1変数 | 一定2変数 |
| --- | --- | --- |
| 0 | [PNG](constant_1d_seed_0_learning_curves.png) | [PNG](constant_2d_seed_0_learning_curves.png) |
| 1 | [PNG](constant_1d_seed_1_learning_curves.png) | [PNG](constant_2d_seed_1_learning_curves.png) |
| 2 | [PNG](constant_1d_seed_2_learning_curves.png) | [PNG](constant_2d_seed_2_learning_curves.png) |

横軸は訓練エピソード／候補数。各曲線257点は対応する `runs/<condition>/<method>/seed_<n>/curve.json` の記録値。元のseed 0用ファイル名も保持した。
