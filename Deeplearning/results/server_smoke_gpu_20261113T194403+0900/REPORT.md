# 正弦波PPO GPU動作確認記録

2026-11-13、サーバs34。**PASS**。このフォルダは動作確認の軽量記録として保存する。

## 実行条件と結果

- PPO、seed 0、1環境、正確に4096学習環境ステップ。16ロールアウト、80 epoch、Adam更新320回。評価は別計上で70環境ステップ。
- MLP 32×32、学習率0.0003、rollout 256、batch 64、5 epoch。gamma 0.99、GAE lambda 0.95、clip 0.2。詳細は `config.json`。
- 既存SineEnvと音響報酬。48 kHz、0.35秒、制御10 ms。訓練は既定のランダムglide、評価は固定1000→2000 Hz、deterministic=True、seed 0。`sine_config.json` に保存。
- 実デバイス: cuda:0、物理GPU 0 / NVIDIA RTX 6000 Ada Generation。学習時間 13.614秒。
- 学習完了。4096報酬等、全320 backward損失と勾配・パラメータ・Adam状態の検査でNaN/無限大なし。
- モデル保存、別プロセスでのGPU再読み込みと35ステップ評価に成功。音声・行動・報酬は再読み込み前後で完全一致。
- 最終音響誤差 0.10508411。動作確認の合否と学習性能は分けて扱い、精度改善は行っていない。

## 残した証拠

`summary.json`、`config.json`、`sine_config.json`、`environment.json`、`environment-freeze.txt`、
`evaluation.json`、`artifact_audit.json`、`source_sha256.json` を保持。
代表図は `frequency_amplitude.png` と `spectrogram.png`。
GPUの再評価差は reload_comparison.json、再読み込み環境は reload_environment.json、実GPU使用は gpu_during_training.json、コード変更ハッシュは source_changes.json。
当時の環境JSON内のPythonパスは履歴として保持している。

## 環境移行と整理

独立した `Deeplearning/.venv-rl` を同じパッケージ版で再構成し、import・CUDA認識・GPU計算と、
この保存済みモデルの35ステップ評価に成功した。元の音声・行動・音響誤差との差はすべて0。
移行確認では追加学習・更新0回。検証時のモデルSHA256:
`10fc39e8b755f932a784ef18af943dbe30e641e1f1eb035e78610d9401ccf849`。

その確認後、テスト用モデル・WAV・重複NPZ・詳細ログ・ソースコピー・キャッシュ・旧仮想環境を
対象2フォルダ内から削除した。モデルと音声の実ファイルはこの軽量記録には含まれない。
全体の移行・削除・容量記録は `Deeplearning/reports/server_smoke_cleanup_20261113/`。

## 今後の再実行

リポジトリ直下の通常SSHターミナルで:

```bash
bash Deeplearning/tools/run_server_smoke.sh cuda:0
```

独立した新環境と現行コードから新しい専用結果を作る。旧テスト内の環境・スクリプトには依存しない。
環境の作成・確認手順は `Deeplearning/envs/rl/README.md`。
GPUが見えないサンドボックス内ではエラーとなる。前回CPU記録のGPU不可はそのアクセス制限であり、共有ドライバ故障の確定ではなかった。
