# 再現・監査

> 2026-11-26 成果物整理：本文の計算結果・実施/未実施は当時の記録です。現在の配置・削除した再生成図・hash検証は[移行記録](../../../../maintenance/20261126/README.md)を参照。旧manifest/hashは原本を維持しています。


リポジトリ直下で実行。出力先はこの正本に固定し、inventoryは既存ディレクトリを拒否、未完了runは上書きしない。新規計算を作り直す場合は、別checkoutで新しい出力パスを設定すること。旧成果は変更しない。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue19-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_protection inventory
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_protection model
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_critics --baseline-only
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_training --workers 9
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_critics
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_report
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_audit --finalize-only
```

保存済み成果の検証だけなら `issue19_audit`、描き直しは `issue19_report`。いずれも再学習しない。旧モデルはnative/model.cppのx0修正により歴史的なsource hashが異なるため、旧の厳密なload_readyでは拒否される。今回のmanifestとmodel_gateで変更を追跡し、旧manifestを改変しない。

各runの `curve.json.gz` は全独立評価/直接法候補、RLの `training_trace.json.gz` は全4096学習サンプル。 `agent.zip` / `selected_agent.zip` と `final.npz` / `selected.npz` を保持。共有の目標音・初期条件・全hyperparameterはmanifestと旧targetへのリンクを参照。NPZはfloat64の元振幅、WAVを重複保存しない。聴取時のみ全音を同じTarget peakで割ってWAVへ出力する。

旧PNGのレンダラは保持一覧に列挙。必要なexact source revisionは旧manifestのgit_head/source_sha256を参照し、旧の数値runを新設定で再学習しない。今回の8図は保存済み波形だけから再生成できる。
