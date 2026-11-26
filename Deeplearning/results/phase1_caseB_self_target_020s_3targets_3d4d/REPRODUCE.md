# 保存済み結果の検証

リポジトリ親から実行する。既存成果の削除・上書きや再学習はしない。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue27g-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27g_report audit
```

このauditは保存WAV・モデル・全source/artifact/Target hash・旧成果・counterを検査するだけで新規音0。事前smokeやanalysis endpointsは音を生成するので、完了済みのledgerがある場合は再実行しない。モデルは推論用で学習再開Buffer/optimizerを含まない。

実行順：prepare stage_a（4音）→3Target図→tests（mapping/標準実装照合/右ε2音/18smoke）→保存モデル読込の引数修正後replay-only（学習再実行なし）→prepare freeze→training（SIGTERM中断）→recovery（同じ54設定・seed、途中成果保護・逐次counter）→analysis physics（保存データのみ）/endpoints（216音）→figures learning→report/freeze/audit。中断の詳細と不確定な追加回数はRECOVERYとinterruption.jsonを参照。
