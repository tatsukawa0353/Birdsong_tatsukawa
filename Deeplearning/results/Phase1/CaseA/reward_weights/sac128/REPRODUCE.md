# 再現・検証

> 2026-11-26 成果物整理：本文の計算結果・実施/未実施は当時の記録です。現在の配置・削除した再生成図・hash検証は[移行記録](../../../../maintenance/20261126/README.md)を参照。旧manifest/hashは原本を維持しています。


リポジトリ直下、既存のDeeplearning/.venv-rlを使う。prepareは既存出力先を拒否し、未完了runも上書きしない。完了済みrunはsourceと成果物hashが合う場合のみ再利用し、全実行済みのtrainは元execution.jsonを保持して終了する。旧成果を更新・再学習しない。

```bash
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue22-mpl
# 本計算前、元ファイルのhash snapshotを保存してから新規prepare
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue22_prepare --before /tmp/birdsong-issue22-before.json
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue22_training --workers 9
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue22_analysis audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue22_analysis critic
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue22_report
```

保存済みモデル/行動の再生はanalysis audit（実再生回数を別記）、鳴管計算なしのチェックはanalysis integrity。描き直し/表の再計算はissue22_report、同じ候補のQ推論だけならanalysis critic。報告段階はいずれも学習しない。

新規実験では、着手前のworkspace数・Git状態・既存研究ファイル/既存コードのpath/bytes/sha256をJSON（storage_before/protected_beforeのschema）に記録し、--beforeへ渡す。別出力先はprepare/trainingの--outputで指定し、解析のOUTは別checkoutでその新規先に固定してからprepareする。元の正本は保持する。

新式のlambdaはスペクトル係数、包絡は1-lambda。旧式のenvelope_weightではない。実効SAC設定/層構造は各runのconfig.json、全4096学習サンプルはtraining_trace.json.gz、33方策評価はcurve.json.gz。元のfloat64波形はfinal.npz/selected.npz、対応重みはagent.zip/selected_agent.zip。posthocは曲線のmin common Eの点・行動診断で、対応する途中modelを追加保存したものではない。

α/分散/entropyは評価曲線のpolicy_statisticsへ保存。固定MC4096点、ローカルseed202622、SB3と同じtanhのepsilon1e-6補正。新しい環境音を生成せず、グローバルRNG/学習状態を変えない。旧条件の温度等は保存Final/Selectedの2点/seedのみ。WAVは全新規音で同じTarget peakの逆数をgainとし、個別正規化やクリップをしない。
