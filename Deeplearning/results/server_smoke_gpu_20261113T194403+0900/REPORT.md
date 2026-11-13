# 正弦波PPO・GPU動作確認

2026-11-13 JST、研究室サーバ `s34`。結果: **PASS**。PPO・seed 0・4096学習環境ステップのGPU実行、保存、独立プロセスでのGPU再読み込み評価を完了した。

## 結果

| 確認事項 | 実測結果 |
|---|---|
| GPU指定 | 物理GPU 0、`CUDA_VISIBLE_DEVICES=0`、プロセス内 `cuda:0` |
| GPU機種 | NVIDIA RTX 6000 Ada Generation |
| GPU UUID | `GPU-d2c4b444-5157-0fa4-1647-8dd22da18ade` |
| 方策・価値ネットワーク配置 | 全13パラメータテンソルが学習前・学習後・再読み込み後とも `cuda:0` |
| backward損失・勾配のデバイス | すべて `cuda:0` |
| 学習環境ステップ | 4096（1環境、117 episode完了＋1ステップ） |
| ロールアウト / epoch / Adam更新 | 16 / 80 / 320 |
| 評価環境ステップ | 学習後35＋別プロセス再読み込み後35＝70（学習には不使用） |
| NaN・無限大 | 検出0件 |
| モデル保存 | `agent.zip`、成功 |
| 独立プロセスでのGPU再読み込み評価 | 成功、訓練PID 1166001、再読み込みPID 1166187 |
| 再読み込み前後の重みハッシュ | 完全一致 |
| 再読み込み前後の音声・行動・報酬・制御系列 | 完全一致、最大絶対差・平均絶対差・RMSEすべて0 |
| 再読み込み前後の音響誤差 | ともに0.10508411191207、差0 |
| 学習時間 | **13.614秒**（前後でCUDA同期） |
| モデル保存時間 | 0.0106秒 |
| 別プロセス起動・再読み込み・評価・図等保存 | 6.249秒 |
| 実行開始〜完了（import含む時刻差） | 23.923秒 |
| 学習中のPyTorch最大GPU割り当て | 18193408 bytes（約17.35 MiB） |

`gpu_during_training.json` のnvidia-smiでも、訓練PID 1166001が上記UUID上で532 MiBを使用していることを確認した。GPU 1は12 MiB・利用率0%で、この実行のcompute processは表示されなかった。PyTorch最大割り当て値はテンソル用の値で、ドライバ・CUDA context等を含むnvidia-smiの使用量とは異なる。

## 前回から変更していない条件

- 前回と同じ仮想環境 `/home/tatsukawa/work/Birdsong_tatsukawa/Deeplearning/results/server_smoke_20261113T190418+0900/.venv`、Python 3.10.12、PyTorch 2.5.1+cu121、SB3 2.8.0、Gymnasium 1.2.3。依存パッケージの追加・変更なし。
- PPO、seed 0、MLP 32×32、学習率0.0003、rollout 256、batch 64、5 epoch、gamma 0.99、GAE lambda 0.95、clip 0.2、entropy係数0、value係数0.5、勾配clip 0.5。残る実効設定も保存済み。
- 既存 `SineEnv` と音響報酬。48 kHz、0.35秒、制御周期10 ms、35ステップ/episode、訓練は既定のランダムglide、評価は固定1000→2000 Hz、振幅0.36–0.6、seed 0、deterministic=True。
- `artifact_audit.json` で、前回 `config.json` のデバイス・説明文・学習済み重みハッシュ以外の全項目、`sine_config.json` 全体、および主要パッケージ版が一致することを照合した。報酬・モデル構造・ハイパーパラメータ探索・学習予算の変更なし。
- NumPyの環境シミュレーションと音声処理はCPUで実行し、方策・価値ネットワークの計算と更新をGPUで実行した。CPU側は従来同様PyTorch intra/inter-op各1スレッド。

## CPU固定箇所と変更内容

事前に `server_smoke.py` と前回 `REPORT.md` を確認し、`experiments.make_agent` の `device="cpu"`、評価時の `PPO.load(..., device="cpu")`、GPU未使用固定の要約を特定した。

1. `server_smoke.py` に `--device cpu|cuda:N` を追加。**既定はcpu**。GPU指定は明示的な番号を必須とし、`auto` や番号省略は受け付けない。
2. `experiments.make_agent(..., *, device="cpu")` を追加し、従来の呼出元ではCPU既定を維持。今回のGPU実行のみ指定デバイスを渡す。
3. CUDA可否・GPU番号を学習前に検査し、指定GPUへ実際にテンソルを割り当てて同期。利用不可時は環境と例外を記録してエラー終了する。SB3の自動CPU切り替えへ進ませない。
4. 方策・価値ネットワーク全パラメータの配置を学習前後・再読み込み後に検査。各backward損失・勾配・更新後パラメータのデバイスも検査する。
5. 別プロセスにも同じ `--device` とGPU可視性を引き継ぎ、GPU上で `PPO.load` する。
6. 再評価差は各配列の最大絶対差・平均絶対差・RMSEと音響誤差差を `reload_comparison.json` に保存。GPUでは完全一致を前提にせず差を記録し、CPU既定時は従来の完全一致検査を維持する。補助 `allclose` の閾値はrtol=1e-5、atol=1e-6で、今回はすべて差0だった。
7. GPU番号・機種・UUID・可視性・パラメータ配置・GPUメモリと、CUDA同期を含む学習時間を保存。異常時の `failure.json` / `reload_failure.json` と環境記録を追加。

変更前ソースは `before_server_smoke.py`、`before_experiments.py`、変更差分は `changes.patch`、変更前後ハッシュは `source_changes.json`、実行時ソースは `source_snapshot/`。

## 実行環境のアクセス制限

| 実行環境 | CUDA認識 | 結果 |
|---|---|---|
| Codexサンドボックス内、同じ仮想環境 | False、GPU数0、nvidia-smi return code 9 | CUDA指定を拒否してエラー終了、CPU学習なし |
| サンドボックス外、同じ仮想環境 | True、GPU 0のみ可視（意図したマスク） | GPU学習と別プロセスGPU評価に成功 |

通常SSHと同じデバイスアクセスを得られるサンドボックス外で実行した。nvidia-smiはホスト上のRTX 6000 Adaを2台表示し、`CUDA_VISIBLE_DEVICES=0` によりPyTorch側では1台だけを可視化した。制限環境の証拠は `restricted_probe/environment.json`、`restricted_probe/failure.json`、`restricted_probe.log`、成功環境は `environment.json` と `reload_environment.json`。

**前回のGPU不可という記録はサンドボックス内での観測であり、ホストの共有ドライバ故障を確定するものではなかった。** 今回、同じ仮想環境でアクセス制限の違いを確認した。共有ドライバ、デバイス権限、共有設定の変更、サーバ再起動は行っていない。

## 検査と保存物

- 全4096報酬・観測・行動・価値・log probability、16ロールアウトのadvantage/return、320 backward損失と全更新の勾配・パラメータ・Adam状態、176件の記録指標にNaN/無限大なし。
- 報酬範囲 -0.77433145〜-0.02058665、backward損失範囲 0.74949515〜8.25492764。
- `cpu_default_check.json`: デバイス未指定のPPOが実際にCPU上に構築され、環境ステップ0・更新0であることを確認した。追加のCPU学習は実行していない。
- `training_transitions.json`、`minibatch_losses.json`、`optimizer_steps.json`、`training_metrics.jsonl`、`progress.csv/json`、`execution.log`、`reload.log` を保存。
- `agent.zip`、`config.json`、`sine_config.json`、`environment.json`、`reload_environment.json`、`environment-freeze.txt` を保存。
- `before_reload.*`、`evaluation.*`、`reload_comparison.json`、`parameter_devices_before_training.json`、`parameter_devices_after_training.json` を保存。
- 目標音 `target.wav` と生成音 `generated.wav`（48 kHz、float32 mono、16800 samples/0.35秒）、`frequency_amplitude.png` と `spectrogram.png` を保存。WAV書き込み後に全sample一致、画像形式・サイズも検査。
- 既存2236ファイル（意図したソース2件は変更後を基準）のSHA256を前後照合し、意図しない変更・削除0。変更前の全台帳も `prechange_original_files_sha256.json` に保持。前回のCPU結果・レポートは変更していない。

## 再実行コマンド

通常のSSHターミナルでリポジトリ直下から実行する:

```bash
bash Deeplearning/results/server_smoke_gpu_20261113T194403+0900/rerun.sh
```

毎回、新しい `Deeplearning/results/server_smoke_gpu_rerun_日時_乱数/` を作成し、GPU 0・PPO・seed 0・4096学習ステップ＋70評価ステップを実行する。標準出力・標準エラーは新出力先の `execution.log`。CUDAが見えない環境ではエラーとなり、CPUには切り替えない。

直接実行する場合（`--output` は新しい専用ディレクトリを指定）:

```bash
CUDA_VISIBLE_DEVICES=0 /home/tatsukawa/work/Birdsong_tatsukawa/Deeplearning/results/server_smoke_20261113T190418+0900/.venv/bin/python -u -m Deeplearning.server_smoke --device cuda:0 --output Deeplearning/results/新しいGPU動作確認名
```

CPU既定を使う既存コマンドでは `--device` を省略できる。

## 残る事項

GPU学習・保存・GPU再読み込み評価に未解決の実行障害はない。サンドボックス内のデバイスアクセス制限は残るため、そこでCUDA指定した場合は記録してエラー終了する。

SB3は小さなMLPのPPOに対してGPU利用率や速度に関する警告を出す。今回の有限値検査もCUDAとの同期を多数含み、時間はこの動作確認条件での実測である。高速化・CPU/GPU速度比較・手法比較は行っていない。

音響誤差は0.10508411で、模倣精度の改善は今回の目的・合格条件ではない。前回CPU方策との一致やCPU/GPU間の学習性能・乱数系列の同一性も要求・判定していない。完全一致を確認したのは今回の同じGPU学習済み重みの再読み込み前後である。長時間学習、追加seed、精度改善は行っていない。
