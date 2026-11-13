# CPU/GPU動作確認結果の整理・環境移行

2026-11-13 JST。**完了**。新しい学習環境を検証してから、指定されたCPU/GPUの2結果内の不要データを削除した。
Gitの追跡済み削除は通常の作業ツリー差分として残している。コミット・push・履歴書き換えは行っていない。

## 容量（duの割り当て量）

| 対象 | 整理前 | 整理後 | 削減量 |
|---|---:|---:|---:|
| CPU動作確認 | 87.00 MiB | 0.31 MiB | 86.68 MiB |
| GPU動作確認 | 63.71 MiB | 0.33 MiB | 63.38 MiB |
| 合計 | 150.71 MiB | 0.64 MiB | **150.07 MiB** |

移行先 `Deeplearning/.venv-rl` は **39.26 MiB**。
新環境を含めた差し引き削減は **110.80 MiB**（この整理記録と新しい手順ファイルの容量は別）。
1 MiB = 1024² bytes。Git履歴の既存オブジェクトは削除しておらず、この数値は対象作業ディレクトリの容量である。
正確なbytes値は `capacity.json`。

整理記録・環境仕様・検証/再実行ツールも含めた実測の純削減は約 **110.20 MiB**。

整理前の主な内訳:

- CPU結果: 旧仮想環境24.45 MiB、Pythonキャッシュ60.48 MiB、Matplotlibキャッシュ0.125 MiB、ソースコピー0.074 MiB、その他約1.86 MiB。
- GPU結果: Pythonキャッシュ61.01 MiB、制限環境probe約0.137 MiB、Matplotlibキャッシュ0.125 MiB、ソースコピー0.082 MiB、その他約2.36 MiB。
- `before_usage.json` と `before_inventory.json` に測定値・分類・シンボリックリンク一覧を保存。

## Gitと対象範囲

開始時の `git status --short` は空で、既存の未コミット変更はなかった。
調査・削除対象は以下の2ディレクトリだけ:

- `Deeplearning/results/server_smoke_20261113T190418+0900`
- `Deeplearning/results/server_smoke_gpu_20261113T194403+0900`

対象外の既存追跡ファイル2333件をハッシュ照合し、意図した `Deeplearning/.gitignore` の1行追加以外の変更なし。
他の実験結果、実音声、既存のリポジトリ直下 `.venv`、学習コード・報酬・ハイパーパラメータは変更していない。
新規追加は独立環境の仕様・手順、検証ツール、この整理記録のみ。

## 新環境

- Python: `/home/tatsukawa/work/Birdsong_tatsukawa/Deeplearning/.venv-rl/bin/python`
- Python 3.10.12、`system-site-packages=True`。旧環境を移動せず、新規venv作成後に依存パッケージを再インストールした。
- 旧環境と同じ共有側のtorch 2.5.1+cu121、NumPy 2.2.6、SciPy 1.15.3、Matplotlib 3.10.8を参照。
- venv内の追加分はSB3 2.8.0、Gymnasium 1.2.3、SciencePlots 2.2.2、cloudpickle 3.1.2、Farama-Notifications 0.0.6。
- `Deeplearning/envs/rl/requirements-local.txt` と `system-constraints.txt` に版を固定。
- `Deeplearning/.gitignore` で `.venv-rl/` を除外。`git check-ignore` でも確認。
- `create.sh` は既存同名環境を上書きしない。今回も作成後に再実行して、変更せずexit 1となることを確認した。
- 主なライブラリの版はCPU/GPUテスト時の記録と一致。`environment-freeze.txt`、`environment-local-freeze.txt` に実行時の全依存と追加分を保存。

## 削除前の動作確認

通常SSH相当のGPUアクセスで、新Pythonを使って検証した（`validation.json`）。

- torch、SB3、Gymnasium、NumPy、SciPy、Matplotlib、SciencePlotsのimport成功。
- CUDA認識True。GPU 0のRTX 6000 Adaを `CUDA_VISIBLE_DEVICES=0` で指定。
- GPUテンソル上の行列積成功。参照計算との最大絶対差0、結果は有限。
- CPUテストのPPOをCPUに読み込み、35ステップ評価成功。元の音声・行動・音響誤差との差は0。
- GPUテストのPPOを `cuda:0` に読み込み、全パラメータの配置を確認し、35ステップ評価成功。元の音声・行動・音響誤差との差は0。
- 評価前後でモデル重み・カウンタ不変。**追加学習0ステップ、更新0回、評価合計70ステップ**。
- 検証時のcheckpoint・policyハッシュは `validation.json` に保存。

削除後にも新環境だけでimport・CUDA認識・GPU行列積が成功した（`post_cleanup_validation.json`）。
旧仮想環境を削除した後も動作し、旧パスへの実行上の依存がないことを確認できた。

## 残した軽量記録

CPU/GPUそれぞれの元ディレクトリに合計30ファイルを残した:

- `REPORT.md`、`summary.json`、`config.json`、`sine_config.json`
- `environment.json`、`environment-freeze.txt`、元の依存版記録
- 代表比較図 `frequency_amplitude.png`、`spectrogram.png`
- `evaluation.json`、`artifact_audit.json`、`source_sha256.json`
- GPUには再評価差・再読み込み環境・実GPU使用のJSONとコード変更ハッシュも保持。

`REPORT.md` は軽量記録として書き直し、現在の環境・再実行手順に更新した。
summary/config/環境JSON/図等の保持ファイルは内容を変えずに残した。
古い環境JSON内のPythonパスは履歴であり、現在の実行用パスではない。

## 削除したものと安全確認

64個の最上位エントリを削除。詳細ログ、冗長なハッシュ台帳、重複NPZ、ソースコピー、テスト用PPO zip、WAV、Python/Matplotlibキャッシュ、旧 `.venv`、旧環境参照のrerun.shが対象。

- 削除前に新環境の検証結果を確認し、移行成功時のみ進めた。
- 削除対象の絶対パスを上記2ルートに限定し、ルート・親ディレクトリがシンボリックリンクでないこと、計画と現在の子一覧が一致することを検査。
- 旧環境で実行中のPythonプロセスがないことを確認。
- シンボリックリンクはリンク自体だけを扱い、リンク先をたどらない。ディレクトリ削除にはPythonのfdベースでリンク攻撃に耐えるrmtreeを使用した。
- 保持ファイルのSHA256を削除前後で照合。対象外ファイルは `preservation_check.json` に結果を記録。
- 絶対パスの計画は `cleanup_plan.json`、実際の削除一覧は `cleanup_result.json`。

## 今後のコマンド

リポジトリ直下の通常SSHターミナルから:

```bash
source Deeplearning/.venv-rl/bin/activate
bash Deeplearning/tools/run_server_smoke.sh cpu
bash Deeplearning/tools/run_server_smoke.sh cuda:0
```

上の2つの実行コマンドはそれぞれ新規出力先を作り、PPO・seed 0・4096学習環境ステップの動作確認を行う。
CPU/GPUの旧結果フォルダや削除済み旧環境には依存しない。
Python bytecodeの生成を抑止して、キャッシュの再肥大化を避ける。
通常のPython実行には `Deeplearning/.venv-rl/bin/python` を使う。

学習なしの環境確認と、環境を新規作成する手順は `Deeplearning/envs/rl/README.md`。

## 残る制約

- Codexサンドボックス内ではGPUが隠される場合がある。通常SSH相当でGPU検証済み。共有ドライバ・共有設定の変更や再起動は行っていない。
- システム由来の `PyNaCl 1.5.0 requires cffi` は元環境と同じくpip checkに残る。今回のRLのimport・GPU計算・PPO評価は正常で、共有環境の修復は行っていない。
- 新venvもシステムパッケージを参照するため、将来の共有側更新の影響は受ける。環境仕様と実測版は保存した。
- テスト用checkpointと音声は指定どおり削除済み。元runの再読み込みをもう一度行う場合はGit履歴から個別取得するか、独立した手順で新規動作確認を行う。
