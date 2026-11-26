# 再現・検査

> 2026-11-26 成果物整理：本文の計算結果・実施/未実施は当時の記録です。現在の配置・削除した再生成図・hash検証は[移行記録](../../../../maintenance/20261126/README.md)を参照。旧manifest/hashは原本を維持しています。


リポジトリ親から同じ .venv-rl を使う。既存成果先は削除・上書きしない。新規計算を再実行する場合は別の成果先と事前計画が必要。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue27f-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27f_report audit
```

これは保存済みデータ・モデル・ソース・WAV・旧成果hashの検査のみで新規音0。事前smokeと36独立詳細再生は既に実行済み。test_issue27fやanalysis endpointsの再実行は新しい音を発生させるため、保存済みのledgerがあれば拒否する。

実行した順序：issue27f_prepare scan → 物理表示の確認 → select --candidate b3_c00（理由指定） → test_issue27f（標準実装照合と3手法128step） → prepare freeze → training --workers9 → analysis physics / figures physics → analysis endpoints（36音） → figures learning → report report / freeze / audit。最初のsynthetic逆変換検査停止と修正後の検査を別ログで保持した。

manifestのcore hashesは学習前に凍結。implementation_checksumsは終了後の分析/描画/報告を含む。artifact_checksumsは自分自身とstorage_afterを除く全成果。推論.ptにはoptimizerやBufferがないため学習再開とは区別する。旧①9音の比較は保存NPZ読込のみ。
