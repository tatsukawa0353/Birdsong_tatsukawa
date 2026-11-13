# 研究室サーバの学習環境

実行環境は `Deeplearning/.venv-rl`。結果ディレクトリから独立しており、Git管理対象外。
既存のリポジトリ直下 `.venv` と共有設定は変更していない。

Python 3.10.12の `--system-site-packages` 環境で、サーバ既存のPyTorch
2.5.1+cu121、NumPy 2.2.6、SciPy 1.15.3、Matplotlib 3.10.8を参照する。
仮想環境に追加する5パッケージは `requirements-local.txt`、共有側の必要な版は
`system-constraints.txt` に固定した。別サーバにそのまま持ち運べる完全独立環境ではない。

リポジトリ直下から実行する。

```bash
# 以後のPythonコマンドで使う環境
source Deeplearning/.venv-rl/bin/activate

# CPUの4096ステップ動作確認
bash Deeplearning/tools/run_server_smoke.sh cpu

# 物理GPU 0の4096ステップ動作確認
bash Deeplearning/tools/run_server_smoke.sh cuda:0
```

どちらもPPO・seed 0、4096学習ステップ＋70評価ステップ。毎回新しい結果フォルダを
作成し、既存結果を上書きしない。GPU引数の番号は物理GPU番号であり、選択した1台を
`CUDA_VISIBLE_DEVICES` で可視化し、プロセス内では `cuda:0` として使用する。
Pythonのbytecodeキャッシュは作成しない。CUDAが見えなければエラー終了し、CPUへ切り替えない。

学習せず、import・GPUテンソル計算だけを確認する場合:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 Deeplearning/.venv-rl/bin/python \
  -m Deeplearning.tools.verify_rl_environment \
  --output /tmp/rl-environment-check-$(date +%Y%m%dT%H%M%S).json
```

通常SSHではGPUを認識する。Codexサンドボックス内ではGPUが隠される場合があるため、
その場合は同じPythonを通常SSH相当のアクセスで実行する。共有ドライバの変更は不要。

環境を新規作成する場合:

```bash
bash Deeplearning/envs/rl/create.sh
```

同名環境があれば上書きせずエラー終了する。既存の共有パッケージ版を照合してから、
固定した追加分をインストールする。ネットワークが必要な場合がある。
共有パッケージが将来更新された場合は、版の不一致を解決してから使用する。

移行時のCPU/GPU保存モデル評価、環境版、容量と削除範囲は
`Deeplearning/reports/server_smoke_cleanup_20261113/` に記録している。
その記録内の旧checkpoint・旧Pythonパスは当時の検証対象の履歴であり、現在の実行には不要。
