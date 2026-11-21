# 再現・検証

リポジトリ直下、既存のDeeplearning/.venv-rlを使用。専用出力先を新規作成して18runのみを学習し、Issue17–19へ書き込まない。prepareは既存manifest/runを拒否する。未完了runも上書きせず、完了済みrunはsourceと全成果物hashが一致するときのみ再利用。全計算完了後のtrainはexecution.jsonを保持して終了する。

```bash
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue21-mpl
# 新規実験のための着手前snapshotが別途必要。既存正本でprepare/trainを繰り返さない。
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue21_training --workers 18
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue21_critics
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue21_audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue21_report
```

保存済み成果の再生検証はissue21_audit（48音を別途計数）、シミュレーションなしの検証は `issue21_audit --integrity-only`。保存音・CSVのみからの描画はissue21_report。critic診断は学習済み重みとIssue19の既存1005候補だけを読み、再学習・鳴管計算を行わない。数値履歴はcurve.json.gzの33独立評価とtraining_trace.json.gzの全4096学習行動を保持。

新しい別出力先の本計算を行う場合は、元ファイルのhash snapshot（protected_before.csv/storage_before.jsonのschema）を用意し、issue21_prepareの--before/--outputおよびissue21_trainingの--outputを同じ新規先へ指定する。解析CLIはこの正本に固定されるため、別出力の解析にはissue21_data.OUTをその別checkoutで変更してからprepareする。元の正本は変更しない。

各runのconfig.jsonに実効設定・Actor/Q1/Q2/targetの層構造、summary.jsonに予算・評価回数・wall/CPU時間・モデル/波形/履歴hashを保存。Final/Selectedを別のagent.zip/selected_agent.zipとNPZで保持。旧32幅のmodel/source hashは当時のものを維持し、今回のnative一致はpreflightとmanifestで追跡する。
