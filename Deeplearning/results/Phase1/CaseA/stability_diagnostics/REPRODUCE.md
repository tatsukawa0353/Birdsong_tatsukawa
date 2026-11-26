# 再現と保存成果の利用

> 2026-11-26 成果物整理：本文の計算結果・実施/未実施は当時の記録です。現在の配置・削除した再生成図・hash検証は[移行記録](../../../maintenance/20261126/README.md)を参照。旧manifest/hashは原本を維持しています。


リポジトリrootで環境を指定する。元成果を保持するため同名のprepareは拒否し、完了runはhashが同じ場合だけ再利用する。

```bash
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue23-mpl
# 着手前のworkspace/git/既存全path・bytes・sha256 JSONを--beforeへ渡す
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue23_stage_a --before /tmp/birdsong-issue23-before.json
# REPORTのStage A・必要理由・事前固定した追加記録を確認した後のみ
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue23_training --workers 3
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue23_analysis audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue23_analysis critics
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue23_analysis local
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue23_report
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue23_analysis integrity
```

新規出力を持つ再現は別checkoutでissue23_data.OUTを専用の新規先へ設定し、着手前JSONを作る。既存正本を上書きしない。学習source/native/target/Stage A/候補hashを固定し、条件変更時は新規studyにする。

既存結果の図/表はissue23_reportで再生成する（鳴管計算/学習0）。criticsは保存Qの集計だけ。localは完了hashがあれば36音を再計算しない。auditは保存重みの再ロードと8個の異なる行動の音を再生するため、実施回数を別に記録する。integrityは音を生成せず旧ファイル/hash/linkを検証する。

保存した.ptはactorとonline twin criticだけの推論用重みで、optimizer/Replay/target Criticを含まず、再開学習はできない。issue23_data.load_checkpoint(path,seed,manifest['settings'])で同じ構造を作って重みをロードし、agent.predict(obs,deterministic=True)で実行できる。posthoc.ptは共通Eによる事後選別で、selected.ptのminJとは別。

全33評価はcurve.json.gz、全4096学習サンプルはtraining_trace.json.gz、実際の4080 Actor更新の前後出力とlossはupdates.json.gz、99固定候補への33時点のQはq_history.json.gz。1更新ごとの音響損失を新たに計算したログではない。actionは正規化[-1,1]、f0_lはHzでなく開閉外力係数。

聴取WAVは全音共通のTarget peak逆数gain、元振幅float64はNPZで保持する。固定MC seed220026・4096標準正規サンプルでstd/entropyを求め、学習RNGには触れない。
