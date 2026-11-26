# Issue27の検証・再現

> 2026-11-26 成果物整理：本文の計算結果・実施/未実施は当時の記録です。現在の配置・削除した再生成図・hash検証は[移行記録](../../../../maintenance/20261126/README.md)を参照。旧manifest/hashは原本を維持しています。


既存3音を保存したまま、以下は新たな音や学習を生成せず検証できる。リポジトリrootから実行する。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue27-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27_report --integrity
Deeplearning/.venv-rl/bin/python -m Deeplearning.tests.test_issue27
# 保存済み3音・native統計からCSV/WAV/図/報告を更新。元Targetは再生成しない
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27_report
```

再生成には別checkoutでissue27_data.OUTを未使用pathへ設定し、issue26_inventory.inventoryの保護集合と同じ全tracked/既存results/reportsをhash化した開始snapshotを/tmp/birdsong-issue27-before.jsonへ用意する（root README/.gitignoreは索引編集用に除く）。

```bash
# 事前3音。ゲート不合格は終了code2となり、本学習へ進めない
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27_prepare physics
# 不合格の診断を確定。現行仕様ではSAC学習コマンドを実装・実行していない
Deeplearning/.venv-rl/bin/python -m Deeplearning.tests.test_issue27
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27_prepare seal
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27_report
```

native/build.jsonにcompiler/flags/ソースと全libraryのhashを保存。元build/libbirdsong.soは変更せず、現行ソースの別.soを生成してbyte一致を確認する。診断probeは既存libraryの実Modelを読み、渡された元bird_advance関数ポインタで積分する。native/*.soは既存Git ignoreの対象なので別環境では同じcompiler/flagsで再ビルドし、保存hashと一致するものを使用する。既存ビルドを上書きして検証しない。

表示STFTは10MHzへの表示用再サンプリング、Blackman-Harris245760/overlap184320、完全窓、0–10kHz。新情報は増えない。波形/WAV/5ms RMSは全てB Target peakの共通gain、PSDもB Target全帯域最大を基準とする。状態traceはnativeを200刻み抽出した可視化用で、統計は10MHz全刻み。変位/速度のstdはドリフトを含む。直線除去は診断統計にのみ使い、音や報酬を変更しない。
