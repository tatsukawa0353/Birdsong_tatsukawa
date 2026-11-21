# 再現と帰省中のオフライン確認

リポジトリrootで、環境を指定する。既存正本のprepareは拒否し、完了runはhash一致時だけ再利用する。

```bash
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue24-mpl
# 着手前のworkspace/git/既存全path・bytes・sha256 JSONを用意
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue24_prepare --before /tmp/birdsong-issue24-before.json
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue24_training --workers 6
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue24_analysis audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue24_analysis critics
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue24_report
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue24_analysis integrity
```

新規再現は別checkoutでissue24_data.OUTを専用の新規先にし、着手前JSONを作る。過去の成果物は上書きしない。Actorだけ独立LRにするためissue24_agent.make_agentを使う。共通learning_rateを変更する方法は禁止。SAC.trainは標準実装、_update_learning_rateだけでActor LRを適用する。optimizer_lrsを各stepで検査し、SB3の共通train/learning_rateログをActor LRと誤読しない。

学習せずに確認する場合、REPORTと9図、CSVとWAV ZIPを開くだけでよい。全33評価はpolicy_evaluations.csvと各runのcurve.json.gz。全1更新の差分/物理量/lossはactor_updates.csv.gz、温度/std/log_stdはpolicy_dynamics.csv、3実効LRはoptimizer_learning_rates.csv.gz。保持率・未到達・Replayの成功はstability_and_acquisition.csv。WAV索引のraw_audio_pathは既存基準も含むため、リポジトリのresultsを一緒に持ち出すかlistening.zipを使う。新たな学習や鳴管計算は不要。

issue24_reportは保存データからCSV/図を再生成するだけで、鳴管計算/学習0。criticsは保存済みQの集計で0音。auditは27checkpointを再ロードし、異なる行動だけを独立再生するため実施回数を別に記録する。integrityは音を生成せず旧成果/hash/link/WAV共通gainを確認する。

checkpoint_index.jsonが全27のpolicy/audio/hashを示す。新規runはfinal.pt/selected.pt/posthoc.ptと対応NPZ、基準はIssue23を参照。issue23_data.load_checkpoint(path,seed,manifest['settings'])で同じMLPを作ってActor+online Critic重みを読み、agent.predict(obs,deterministic=True)で使用できる。loaderのoptimizerは推論用の仮初期化で、学習LR・optimizer状態の復元ではない。optimizer/Replay/target Criticを含まず再開学習はできない。metadataのformatは共用Issue23 reader形式を指し、今回の条件はconfig.json/manifestを正本とする。

WAVはTargetを含む全音共通gain=1/Target peakのfloat32で、クリップや個別正規化はない。音響評価と元波形はfloat64 NPZに保持する。f0_lはHzの基本周波数ではなく開閉外力係数。Selected=minJ、posthoc=min共通E、Final=最終更新後を混同しない。
