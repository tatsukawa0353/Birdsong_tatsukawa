# 研究室サーバ・正弦波RL動作確認

2026-11-13 JST、サーバ `s34`。結果: **PASS**。動作確認のみを行い、学習性能・手法の優劣は判定しない。

## 手法・設定

- PPO、seed 0、1環境、学習予算は正確に4096環境ステップ。既存 `experiments.make_agent` の256ステップのロールアウトが4096を割り切り、CPUで短時間に実行可能なため選択した。
- MLP 32×32、学習率0.0003、batch 64、1ロールアウト当たり5 epoch、gamma 0.99、GAE lambda 0.95、clip 0.2、entropy係数0、value係数0.5、勾配clip 0.5。残る実効値は `config.json` と `agent.zip` に保存。
- 既存 `SineEnv`、48 kHz、0.35秒、制御周期10 ms、35ステップ/episode、周波数500–3000 Hz、振幅0–1、既存の音響報酬。訓練目標は既存既定のランダムglide。評価は固定1000→2000 Hz、振幅0.36–0.6、deterministic=True、reset seed 0。
- `sine_config.json` は訓練・評価の環境設定、`config.json` は実行設定、`source_snapshot/` と `source_sha256.json` は実行コードの保存。

## ステップと更新回数

| 種類 | 実測 |
|---|---:|
| 学習の環境ステップ | 4096 |
| 完了episode | 117（残り1ステップで学習終了） |
| ロールアウト収集・学習呼出 | 16 |
| epoch数 / SB3 `_n_updates` | 80 = 16×5 |
| ミニバッチ単位のAdam更新 | 320 = 16×5×(256/64) |
| 保存時モデルの評価 | 35 |
| 別プロセス再読み込み後の評価 | 35 |
| 評価専用の環境ステップ合計 | 70 |
| 学習＋評価の全環境ステップ | 4166 |

Adamは方策・価値ネットワークのパラメータを一緒に更新する。80はPPOのepochカウンタであり、4096環境ステップや320ミニバッチ更新とは別。評価の70ステップは学習・更新に使用していない。既存の複数手法比較CLIは使わず、比較・ランキング・追加seed・初期性能評価・途中性能評価は実行していない。

## 確認結果

- 学習が正常終了し、学習前後で重みのSHA256が変化した。
- 4096件の観測・報酬・行動・価値・log probability、16ロールアウトのadvantage/return、全320件のbackward損失と勾配・パラメータ・Adam状態、176件の記録指標に非有限値なし。報酬範囲 -0.78029251〜-0.03083399、ミニバッチ損失範囲 0.80754948〜7.41322994。`training_transitions.json`、`minibatch_losses.json`、`optimizer_steps.json`、`training_metrics.jsonl`、`progress.csv` に記録。
- `agent.zip` 保存成功。独立Pythonプロセスで `PPO.load(..., device="cpu")` し、35ステップの評価成功。再読み込み前後の重みハッシュ・行動・生成音・目標音・報酬・制御系列が完全一致。`before_reload.*`、`evaluation.*`、`reload.log` に記録。
- `target.wav` と `generated.wav`: 48 kHz、16800 samples、0.35秒、mono float32、非有限値なし。独立の音量正規化はせず、元の振幅を保存。書き込み後に読み直して全sample一致を検証した。
- `frequency_amplitude.png`: 目標と再読み込み後方策による正弦波の周波数・振幅を10 ms単位で比較。正弦波生成器に実際に渡した値を表示しており、音声からの周波数推定値ではない。`spectrogram.png` も保存。
- 既存ファイル2200件のSHA256を前後照合し、変更・削除0件。既存学習コード、報酬、共有設定、旧結果、既存仮想環境を変更していない。

## 実行時間と実際のデバイス

- 学習: **6.535秒 wall / 6.482秒 process CPU**。有限値監視を含む。
- モデル保存: 0.0051秒。
- 別プロセス起動＋再読み込み＋評価＋音声/図保存: 5.991秒。
- ライブラリimport後の一連の処理: 18.260秒。
- importを含む実行開始記録〜完了: **25.434秒**（`run_started.json` と `summary.json` のJST時刻差）。環境準備・実行スクリプト作成を含む出力先作成〜完了は285.2秒。
- CPU: **INTEL(R) XEON(R) GOLD 6526Y**、PyTorch intra/inter-op各1スレッド、学習中の観測CPU IDは[17]。スレッド数はプロセス内のみ設定。方策・価値ネットワークの全パラメータは実測 `cpu`。
- GPU使用なし。`torch.cuda.is_available()=False`、device_count=0、`nvidia-smi` はドライバとの通信失敗（return code 9）。CUDA対応ビルドのPyTorchであっても、今回の計算はCPU。GPU名・GPU動作は確認できていない。

## 環境の不足と対応

1. README記載の `Deeplearning/.venv` はこのサーバにはなく、リポジトリ直下 `.venv` はPython 3.10.12、torch/SB3/SciencePlots不足。一方、システムPythonにPyTorch 2.5.1+cu121、NumPy/SciPy/Matplotlibがあった。今回の `/home/tatsukawa/work/Birdsong_tatsukawa/Deeplearning/results/server_smoke_20261113T190418+0900/.venv` を `python3 -m venv --system-site-packages` で作り、大きな既存パッケージを再利用して不足分だけ追加した。
2. 既存torch等を制約ファイルで固定し、互換性のあるSB3 2.8.0、Gymnasium 1.2.3、SciencePlots 2.2.2、cloudpickle 3.1.2、Farama-Notifications 0.0.6を専用環境へ追加。READMEの過去のPython 3.12/torch 2.14.1+cpu固定環境そのものの再現ではない。実環境は `environment.json`、全パッケージは `environment-freeze.txt`、追加分は `runtime-additions.txt`。
3. 専用環境を作った直後のpip 22では `--report` が未対応だったため、そのオプションを除き、インストールログとfreezeで記録。サンドボックス内の最初のネットワーク取得はDNS制限で失敗し、許可されたネットワーク実行で取得成功。失敗分も `environment-setup.log`、`environment-install.log` に保持。
4. Matplotlibがホーム配下へキャッシュを書けないため、この実行の `MPLCONFIGDIR` とPythonキャッシュを専用出力配下に設定。共有設定は変更していない。
5. 既存の学習・報酬コードに動作不具合は見つからず、その修正は不要だった。追加した `Deeplearning/server_smoke.py` は今回用の実行・監視・保存・別プロセス評価処理のみ。

## 再実行

同じサーバで既存の今回専用環境を使い、リポジトリ直下から:

```bash
bash Deeplearning/results/server_smoke_20261113T190418+0900/rerun.sh
```

毎回 `results/server_smoke_rerun_日時_乱数/` を新規作成し、今回の結果を上書きしない。実行中の標準出力・標準エラーは新出力先の `execution.log` に保存。再実行もPPO seed 0、4096学習ステップ＋70評価ステップである。

直接のCLIは次のとおり（`--output` には新しい専用パスを指定）:

```bash
/home/tatsukawa/work/Birdsong_tatsukawa/Deeplearning/results/server_smoke_20261113T190418+0900/.venv/bin/python -m Deeplearning.server_smoke --output Deeplearning/results/新しい専用名
```

## 残る問題と性能の扱い

- GPUドライバは利用不可のまま。今回のCPU動作確認は成功したが、GPU実行は未確認。
- システム由来の `PyNaCl 1.5.0 requires cffi` が `pip check` に残る。元のシステム環境にも同じ欠落があり、RLの使用経路には現れなかったため共有環境を修復していない（`system-pip-check.log`、`pip-check.log`）。
- 今回の環境はシステムパッケージを参照するため、将来それらが更新されると実行環境も影響を受ける。今回実行時点の版は記録済み。
- 固定評価の音響誤差は **0.10414409**。目標の周波数・振幅の追従は不十分で、図でも周波数は約1700 Hz付近に留まる。既存の暫定閾値0.08には未達だが、この値・閾値は動作確認の合否に使っていない。精度改善、報酬変更、探索、他手法/seed比較、追加学習は行っていない。
