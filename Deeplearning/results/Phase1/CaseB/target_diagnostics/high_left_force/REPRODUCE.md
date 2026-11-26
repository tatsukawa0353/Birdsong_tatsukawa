# Issue27改訂Stage Aの再現

> 2026-11-26 成果物整理：本文の計算結果・実施/未実施は当時の記録です。現在の配置・削除した再生成図・hash検証は[移行記録](../../../../maintenance/20261126/README.md)を参照。旧manifest/hashは原本を維持しています。


リポジトリrootから、まず保存済み結果を検査する。下記の検査/再集計は新しい鳴管音や学習を生成しない。実環境は.venv-rl、各BLAS/CPU thread1。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue27r-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27r_report --integrity
Deeplearning/.venv-rl/bin/python -m Deeplearning.tests.test_issue27r
# 保存9音だけから表/図/WAV/報告を更新
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27r_report
```

物理生成を再現する場合は別checkoutでissue27r_data.OUTを未使用pathに変更し、着手前の全tracked（root README/.gitignore除く）＋results/reportsをpath/bytes/sha256でsnapshot化する。JSON形式はstorage_before.jsonにprotected_before.csvの全行をfilesキーとして加えたものと同じ。元成果のhash不一致・既存OUTではscanを拒否する。

```bash
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27r_prepare scan --before /tmp/birdsong-issue27r-before.json
# 全点非持続なら終了code2。Targetも学習も作らない
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27r_prepare select
Deeplearning/.venv-rl/bin/python -m Deeplearning.tests.test_issue27r
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27r_prepare seal
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27r_report
```

native/build.jsonにcompiler/flagsと全hashを記録。元native/model.cpp/libbirdsong.soは変更せず、同じソースを別.soへビルドしてbyte一致を検証する。計測は実libraryのbird_advanceへの関数ポインタで、probeは読み取りと統計のみ。native/*.soは既存Git ignore対象のため、別環境では同じcompiler/flagsで再ビルドし保存hashと照合する。

初回scanの実行ソースはexecuted_sourcesに保持。後の停止guard・報告追加と区別し、scan関数の構文木、native wrapper、現行モデルのhashを検査する。今回の9点として新規生成したc21は、前回の保存B oracleと読み取り比較する。旧ファイルの再作成・書換えや、照合目的の追加生成はしない。

全9点のPSD表示はc00の全帯域PSD最大を共通基準とする。Target未選定でありc00は表示用参照。表示用10MHz再サンプリング、Blackman-Harris245760/overlap184320、完全窓、0–10kHz。WAV/波形/RMSは全9音最大peakによる共通gain。state traceはnativeを200刻み抽出した可視化用で、統計はnative10MHz全刻み。直線除去はドリフト診断のみに使う。
