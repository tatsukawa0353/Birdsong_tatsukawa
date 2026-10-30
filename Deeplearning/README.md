# Birdsong 制御実験

既存のシミュレーション・描画コードを変更せず、Issue #1–#7 の調査・数値 API・音響報酬・強化学習比較をこのフォルダに隔離した。作業結果は [REPORT.md](REPORT.md)、モデル仕様は [model_audit.md](reports/model_audit.md)、実録音の管理は [DATA_POLICY.md](reports/DATA_POLICY.md) を参照。

## 実行環境

以下はリポジトリ直下から実行する。実行済みの専用環境は `Deeplearning/.venv`。既存の `venv/` には変更を加えていない。

```bash
python3 -m venv Deeplearning/.venv
Deeplearning/.venv/bin/python -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
Deeplearning/.venv/bin/python -m pip install -r Deeplearning/requirements-lock.txt
export MPLCONFIGDIR=/tmp/birdsong-mpl
Deeplearning/.venv/bin/python -m Deeplearning.model
Deeplearning/.venv/bin/python -m unittest discover -s Deeplearning/tests -v
```

Python 3.12、g++ が必要。CPU 版 PyTorch の固定版は `requirements-lock.txt` と一致させている。GPU は使っていない。PyTorch の公式 CPU インデックスにその版が必要であり、通常の PyPI だけでは `+cpu` 版を取得できない場合がある。

## 合成音実験

```bash
Deeplearning/.venv/bin/python -m Deeplearning.reproduce
Deeplearning/.venv/bin/python -m Deeplearning.run phase0
Deeplearning/.venv/bin/python -m Deeplearning.run phase1
Deeplearning/.venv/bin/python -m Deeplearning.run phase1 --config Deeplearning/configs/phase1_extended.json --output Deeplearning/results/phase1_extended
Deeplearning/.venv/bin/python -m Deeplearning.diagnostics
```

設定は `configs/`、出力は `results/`。学習曲線・最終評価・初期評価・評価に使った環境呼出回数・実行時間・seed・重み・制御値を保存する。予算は明示的に設定し、PPO のロールアウトでステップ数が丸められた場合も実際の `training_steps` を記録する。現在の Phase 0 では4096が256で割り切れるので超過しない。

SAC/TD3/PPO は32×32 MLP、CPU1スレッド、学習率0.0003。SAC/TD3 はバッチ64、学習開始256、毎ステップ更新、バッファ50000。TD3 の学習探索ノイズは標準偏差0.1。PPO はロールアウト256、バッチ64、5 epoch。残るパラメータはロックした SB3 の既定値であり、保存した `agent.zip` にも含まれる。評価は全手法とも `deterministic=True` で探索ノイズを加えない。

Phase 0 は48 kHz、周波数500–3000 Hz、振幅0–1。目標に現在の周波数・振幅を観測として与える単純な追従課題である。訓練は乱数seedによる異なるグライド、評価は固定の1000→2000 Hz。同じ波形を評価と訓練に固定して暗記させる構成ではないが、分布外の音への一般化を評価したものでもない。

Phase 1 はモデル自身の既知パラメータで作る0.35秒の目標。同じ環境でランダム行動、定数制御、既知制御、定数のランダム探索、差分進化を比較する。差分進化は CMA-ES ではなく、直接最適化の別の候補である。oracle は既知の生成パラメータを使う上限確認で、学習手法ではない。

報酬は共有スケールでの対数 PSD 誤差 + 振幅包絡誤差の負値。必要なら行動変化量の二乗平均に重みを付ける。位相合わせ、時間伸縮、各波形の独立音量正規化は行わない。即時報酬と終端全体指標は別に記録する。終了は有限時間課題の終端 `terminated=True` とし、発散も異常終端として扱う。

## 実録音（ローカル限定）

ユーザーがローカル利用許諾を確認した0.35秒版のみを使う。以下のフラグは確認済み事実を記録するもので、許諾取得を代行するものではない。

```bash
Deeplearning/.venv/bin/python -m Deeplearning.phase2 --target Deeplearning/sample_song/14412_0.35s.wav --local-permission-confirmed
Deeplearning/.venv/bin/python -m Deeplearning.private_analysis
```

結果は Git 対象外の `private_runs/phase2/`。`private_analysis` は保存した制御を再生して速度RMSを測定し、特徴量・聴取用生成音・図・ローカルレポートを作る。A は右の寄与を除外（3自由度）、B は右 `f0=1e7` 固定（4自由度）、C は左右可動（5自由度）。左/右の更新は同時に行う。全条件に同じ録音区間・SAC・seed・予算・評価関数を用いる。左右で可動自由度が異なるので、優劣を器官数だけで説明できない。

新 API や設定を変更して再実行すると既定の結果ファイルは上書きされる。比較用の実験には別の `--output` を指定する。`private_runs` の学習成果物・音声は公開コミットに加えない。

API の参照: [Gymnasium Env](https://gymnasium.farama.org/api/env/)、[SB3 custom environments](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)、[SB3 examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)。
