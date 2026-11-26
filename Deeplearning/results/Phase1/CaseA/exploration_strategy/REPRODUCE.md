# Issue26の再現

> 2026-11-26 成果物整理：本文の計算結果・実施/未実施は当時の記録です。現在の配置・削除した再生成図・hash検証は[移行記録](../../../maintenance/20261126/README.md)を参照。旧manifest/hashは原本を維持しています。


リポジトリrootで実行する。新規研究は新しいcheckoutでissue26_data.OUTを未使用の専用pathに設定し、着手前snapshotを新規保存する。prepareは既存OUTを拒否し、完了runは同じソース/hashのときだけ再利用する。途中失敗/既存成果を上書きしない。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue26-mpl
# 着手前snapshot。既存snapshot/既存OUTを拒否
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue26_inventory
# 6試験とrecord生成。native119音を伴う
Deeplearning/.venv-rl/bin/python -m Deeplearning.tests.test_issue26
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue26_prepare
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue26_training wide --workers 3
# STAGE_B.md確認後、Bの結果にかかわらず全C/Dを実行
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue26_training local --workers 6
# 51音を独立再生。再実行した場合は追加回数を別記
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue26_analysis audit
# 保存データから図/表/WAV/報告を更新。native0
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue26_report
# hash/保護/link/WAV検査のみ。native0
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue26_report --integrity
```

manifestに環境/実効設定/source/native/Target/PLANのhashを保存。checkpointはActor+online Critic推論用でoptimizer/Replay/local RNGを含む再開用ではない。全候補・33評価点・4080更新をlossless gzip保存。凍結済み成果を検証するときは、図や表を再生成する前にintegrityを実行する。
