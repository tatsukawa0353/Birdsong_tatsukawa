# 再現とオフライン確認

リポジトリrootで環境を指定する。新規計算は別checkout/新規OUTと着手前hash snapshotを使い、既存正本のprepareや未完了runの上書きは拒否する。

```bash
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue25-mpl
Deeplearning/.venv-rl/bin/python -m unittest Deeplearning.tests.test_issue25 -v
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_prepare --before /tmp/birdsong-issue25-before.json
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_training --workers 6
# Stage Bが終わってからStage C。offlineは鳴管計算・Actor更新0
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_offline --workers 3
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_analysis audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_analysis critics
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_analysis offline_audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_report
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_analysis integrity
```

prepareの前に、workspace数・git状態・旧results/reports/追跡コードのpath/bytes/SHA256を記録したJSONが必要。共通learning_rateを変更せず、issue25_agent.make_agentを使う。基準3e-4も同じscheduler/標準SAC.trainで比較検証した。実効param_groups LRは全step/evalで検査する。

帰省中はREPORT・10図・CSV・聴取ZIPを開けば追加計算不要。policy_evaluations.csvは各33点、policy_dynamics.csvは温度/std、actor_updates.csv.gzは1更新の行動/物理差とloss、optimizer_learning_rates.csv.gzは3実効LR（既存基準はprovenanceと区別）。各runのcurve/updates/q_historyには保存後に復元できない中間診断がある。minJ/minE/Final/first_successを混同しない。

offline_data.npzとoffline_batches.npzが固定train/holdoutと全batch列。offline_points.csv.gzが各点の出典/hash、dataset_auditとholdout_nearest_neighborsが重複・近傍・偏りを示す。offlineのseedは初期化とbatch列のseedで、全3source seedのtrainを毎回共有する。holdoutは更新に使わず、全LRとも4080回帰更新。offlineはcritic.ptのみ、Actorを含まない。

オンライン.ptは共用Issue23 reader形式のActor+online twin Criticで、issue23_data.load_checkpoint(path,seed,manifest['settings'])で推論できる。loaderのoptimizerは仮初期化で学習LR/状態の復元ではない。offlineは同じSAC構造のcritic.load_state_dictで読めるが、オンラインActorへ接続する実験は本Issueでしていない。どちらもoptimizer/Replayを含まず再開学習用ではない。

report/tables/critics/offline_auditは学習や鳴管音を生成しない。online auditは保存方策の独立再生を行うので、再実行回数は別に記録する。integrityはhash/link/WAV gainを検査するだけ。WAVは全音共通gain=1/Target peak、元波形はfloat64 NPZ。f0_lはHzでなく開閉外力係数。
