# Issue #16 — Phase 0 最終学習率比較

> 2026-11-26 成果物整理：本文の計算結果・実施/未実施は当時の記録です。現在の配置・削除した再生成図・hash検証は[移行記録](../../../../maintenance/20261126/README.md)を参照。旧manifest/hashは原本を維持しています。


TD3/SAC・seed 0/1/2・λ=0.5・102,400環境ステップを固定し、LR=0.0001の6 runのみ新規実行。LR=0.0003の6 runは保存結果を読み取り専用で利用した。**本比較でPhase 0を終了する。追加探索・PPO再実験はしない。**

## 条件と整合性

0.35秒、48 kHz、35行動×10 ms、1000→2000 Hz glideと既存振幅系列。報酬は局所 `-(E_spec+0.5 E_env)`、共通評価は全区間 `E_spec+E_env`（λ=1）。CPU・各worker PyTorch 1 thread、MLP [32,32]、γ=.99、batch64、buffer50000、warmup256、train_freq=1、gradient_steps=1。TD3 noise=.1、SAC entropy=auto（初期1）。科学計算4ソースと依存版は既存manifestに一致し、実効設定はoptimizer groupsを含めLR以外一致、初期方策hashと256 warmup行動・報酬も一致した。

各runは102,400環境step（2925完了episode＋25部分step）であり、gradient iterationsは102,144。TD3の実optimizer呼出しはactor51,072/critic102,144、SACはactor/critic/entropy各102,144。評価は0＋2048ごとの51 callback点と独立した最終更新後1回の計52音、1820評価step。提示曲線51点の終点は最終更新後で、モデル・音声・表と一致。`callback_curve.json`に更新直前の元の終点も残した。`audit.json`で両者の差とhashを確認できる。

NaN/Infはloss、勾配、更新後パラメータを全更新で検査し0件、数値失敗0件。無音フラグとloss min/maxは各summary/curveに保持。無音や高誤差のrunを除外していない。Oracle・独立評価の学習状態/RNG不変性・保存モデル再読み込み・波形からの評価再計算を検証した。

## 最終方策の3 seed平均 ± 標本SD

| 手法 | LR | E_spec | E_env | E_total | 周波数MAE Hz | 振幅MAE |
| --- | --- | --- | --- | --- | --- | --- |
| TD3 | 0.0003 | 0.0151752 ± 0.00288 | 0.00250722 ± 0.000341 | 0.0176824 ± 0.00297 | 12.0674 ± 7.13 | 0.00199957 ± 0.000429 |
| TD3 | 0.0001 | 0.0162651 ± 0.00277 | 0.00323436 ± 0.00025 | 0.0194994 ± 0.00264 | 20.5484 ± 13.6 | 0.00353574 ± 0.000443 |
| SAC | 0.0003 | 0.0229139 ± 0.00398 | 0.00302848 ± 0.000391 | 0.0259424 ± 0.00437 | 46.2429 ± 10.5 | 0.00319512 ± 0.00105 |
| SAC | 0.0001 | 0.0208589 ± 0.00282 | 0.00309086 ± 0.00046 | 0.0239498 ± 0.00323 | 44.701 ± 5.44 | 0.00295932 ± 0.00113 |

- TD3: LR 0.0001 の平均最終誤差は 0.0194994、既存0.0003との差は +0.00181699。この3 seedでは改善しなかった。
- SAC: LR 0.0001 の平均最終誤差は 0.0239498、既存0.0003との差は -0.00199261。この3 seedでは改善。

これは同一の学習目標音に対する小規模比較であり、汎化性能や有意差を示すものではない。途中最良を最終の代わりに採用しない。

## 個別値・途中最良（補助）・所要時間

| 手法 | LR | seed | 最終 E_total | 途中最良 E_total | 最良step | 学習秒 | 評価秒 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TD3 | 0.0003 | 0 | 0.01446309 | 0.01050316 | 102400 | 277.14 | 1.70 |
| TD3 | 0.0003 | 1 | 0.01827586 | 0.01680548 | 81920 | 279.74 | 1.71 |
| TD3 | 0.0003 | 2 | 0.02030837 | 0.01508368 | 100352 | 275.91 | 1.70 |
| SAC | 0.0003 | 0 | 0.02360683 | 0.01484032 | 24576 | 415.57 | 1.89 |
| SAC | 0.0003 | 1 | 0.03097869 | 0.02207071 | 71680 | 411.85 | 1.89 |
| SAC | 0.0003 | 2 | 0.02324161 | 0.01796722 | 94208 | 419.93 | 1.89 |
| TD3 | 0.0001 | 0 | 0.02202701 | 0.01913449 | 102400 | 304.54 | 1.87 |
| TD3 | 0.0001 | 1 | 0.01970836 | 0.01932985 | 34816 | 301.97 | 1.88 |
| TD3 | 0.0001 | 2 | 0.01676291 | 0.01058499 | 94208 | 302.39 | 1.87 |
| SAC | 0.0001 | 0 | 0.0261866 | 0.0217894 | 102400 | 437.11 | 2.06 |
| SAC | 0.0001 | 1 | 0.02024804 | 0.02024804 | 102400 | 441.45 | 2.08 |
| SAC | 0.0001 | 2 | 0.02541467 | 0.02013498 | 88064 | 441.41 | 2.08 |

新規6 runの並列wall時間 448.54秒、学習process wall合計 2228.87秒、評価合計 11.85秒。保存等は各summaryのtotalと学習/評価との差に含まれる。描画・レポート時間は `report_timing.json` に別記。事前320 step smokeは比較予算外で `smoke.json` に記録。

## 成果物と再検証

- `comparison_per_seed.csv` / `comparison_aggregate.csv` / `comparison_curves.csv`: 全12 run、音響2成分・MAE・best・到達step・費用。
- `figures/learning_curves_seed0.png` / `learning_curves_mean_sd.png`: TD3/SACの2パネル、各パネルは2 LRだけ、生51点、mean±sample SD、平滑化なし。各LR seed0の最終スペクトログラム・物理制御図も保存。
- 新規6 runに最終/bestモデル、初期/最終npz・WAV、実行設定・lossとoptimizer回数・全訓練traceを保存。初期/best/finalの意味は各summary/best.jsonに分離。
- `manifest.json` / `protected_before.json` / `audit.json`: 版・計算ソース・旧成果物全hash。既存変更済みのグラフ2枚も現在のバイト列を維持した。監査 15371 checks passed。

リポジトリ親から `.venv-rl` の絶対Pythonパスで実行する。既存結果がある場合、再学習コマンドは上書きを拒否する。

```bash
cd /home/tatsukawa/work/Birdsong_tatsukawa
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue16-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue16 audit
# 同じ既存baselineを持ち、今回の出力がまだ存在しない作業コピーで:
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue16 prepare
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue16 train --workers 6
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue16 finish
```

Phase1の0.20秒実験は別Issue #17・別環境・γ=1・LR=.0003・全区間報酬で行う。この比較のLRやλは移植しない。

実際のPythonパス・Python/C++コンパイラ版・CPU機種・CPU affinity・描画コードhashは `completion_verification.json` に保存した。GPUは使用していない。
