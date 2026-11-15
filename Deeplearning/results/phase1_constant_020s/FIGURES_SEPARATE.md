# Phase 1（Issue #17）：次元別の独立した誤差線図

Issue #17の保存済み `comparison_curves.csv` の値から**再学習・方策評価なし**で生成した図。1D、2D、3Dをそれぞれ独立した画像にした。元の3段結合PNGは再現性のため保存し、削除・書換えを行わない。

縦軸は各次元の値に合わせる。**次元ごとに自動スケールが違うので，絶対誤差の大小を目で比較する際は縦軸数値を読むこと。** 各図にはTD3/SAC/DE/Randomの4曲線を表示。平滑化・間引きなし。学習途中の最良値と最終性能を混同しない。

## 主図：その時点の誤差

| 条件 | seed 0 | 3 seed平均・標本SD |
| --- | --- | --- |
| **1D**（ε_l） | [1D seed0](figures/learning_curves_1d_seed0.svg) | [1D mean±SD](figures/learning_curves_1d_mean_sd.svg) |
| **2D**（ε_l，p_s） | [2D seed0](figures/learning_curves_2d_seed0.svg) | [2D mean±SD](figures/learning_curves_2d_mean_sd.svg) |
| **3D**（ε_l，p_s，f0_l） | [3D seed0](figures/learning_curves_3d_seed0.svg) | [3D mean±SD](figures/learning_curves_3d_mean_sd.svg) |

主図ではRLはその評価時点の決定論的actor，DE/Randomはその試行で実際に生成した候補を意味する。両者は同じ「途中最良解」ではない。

## 補助図：その時点までに観測した最良値

| 条件 | seed 0 | 3 seed平均・標本SD |
| --- | --- | --- |
| **1D** | [1D seed0](figures/best_observed_1d_seed0.svg) | [1D mean±SD](figures/best_observed_1d_mean_sd.svg) |
| **2D** | [2D seed0](figures/best_observed_2d_seed0.svg) | [2D mean±SD](figures/best_observed_2d_mean_sd.svg) |
| **3D** | [3D seed0](figures/best_observed_3d_seed0.svg) | [3D mean±SD](figures/best_observed_3d_mean_sd.svg) |

この最良値は**学習中に観測した方策/候補の最良値**であり，RLの最終モデルに保証された性能ではない。平均±SDの陰影の下限のみ表示上0にクリップ（誤差やSDの数値そのものは変更しない）。

## 追加の高解像度PNG生成

今後使う場合はGitHubに追加した再描画用コードで別々のPNGも生成可能。既存出力を変更せず新規12枚を作る。

```bash
# リポジトリ直下、main_deeplearningの更新を取得してから
PYTHONDONTWRITEBYTECODE=1 Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17_redraw
```

なお，このSVGは直接保存済みCSVから作成したグラフであり，元の実験レポート・結果のハッシュや数値を変更していない。

- [元のREPORT.md](REPORT.md)
- [元のcomparison_curves.csv](comparison_curves.csv)
