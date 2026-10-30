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

## スペクトログラムの表示設定

`plotting.py` に共通設定を集約した。以前の実録音の描画手順に合わせ、描画時に音声を `target_fs=1e7`（10 MHz）へリサンプリングしてから、`nperseg=245760`、`noverlap=184320`、`window_type='blackmanharris'` をそのまま適用する。窓は24.576 ms、ホップは6.144 ms、重複75%。10 MHzの元波形には再リサンプリングを行わない。アップサンプリングは表示条件の統一であり、元の周波数情報を増やすものではない。

表示は `cmap='gray_r'`、`vmin=-50.0`、`vmax=-20.0`。各図の最大PSDを基準とする相対dBで、周波数の表示帯域は0–10000 Hz。カラースケールと周波数帯域は別の設定である。

描画用STFTにはゼロ埋めを使わず、窓全体が音声内に収まるフレームだけを使う。横軸も最初・最後の有効なフレーム中心で切り、0秒・0.35秒の端を表示しない。0.35秒音声の表示域は0.012288–0.331776秒となる。報酬用の `acoustics.py`、学習済みモデル、保存音声、誤差の数値は変更しない。

学習し直さずに既存の図だけを更新する場合:

```bash
Deeplearning/.venv/bin/python -m Deeplearning.redraw --include-private
```

`--include-private` を省略すると合成実験の図のみ更新する。実録音由来の図は従来どおり `private_runs/` 配下にのみ保存する。

## Issue #10：合成音の報酬診断とPhase 1比較

結果は [ISSUE10_REPORT.md](reports/ISSUE10_REPORT.md)。実録音を読まず、0.35秒のモデル内合成目標だけを使う。旧報酬を既定値として保持し、目標の活動マスク・帯域包絡・相対スペクトル形状・欠落ペナルティによる候補を比較する。新指標と旧誤差の改善を区別する。

```bash
Deeplearning/.venv/bin/python -m Deeplearning.issue10 all
```

段階別は `audit`（固定合成波形・感度・時間集計）、`train`（1050step×3 seed）、`baselines`（oracle/定数/ランダム/DE）、`extend`（事前条件を満たした手法を全条件4200stepで比較）、`glide`（定数目標での再現性に応じてRLまたはoracleのみ）、`report`。設定は `configs/issue10.json`、全成果物は `results/issue10/`。`train`/`extend` は同設定の学習済み結果を再利用する。`report` は既存結果の整合性も再計算する。

環境の選択は `ModelConfig(loss_kind="balanced", reward_timing="context")` または `reward_timing="aligned"`。alignedは新規frame寄与を一度だけ評価し、環境の `reward_discount` を学習器のgammaと一致させる。既定は `loss_kind="baseline", reward_timing="context"`。全条件の最終 `error` は従来の `acoustics.distance`、`objective_error` は新しい全体損失、`mean_reward_cost` と `discounted_reward_cost` は実際のstep報酬の集計。表示の色範囲は報酬設定へ渡さない。

## Issue #9：発声状態と外力制御の診断

対応結果は [ISSUE9_REPORT.md](reports/ISSUE9_REPORT.md)、録音由来の比較表はローカル限定の `private_runs/issue9/REPORT.md`。合成の発声マップは `results/issue9/` に保存した。

```bash
Deeplearning/.venv/bin/python -m Deeplearning.issue9 all --local-permission-confirmed
```

段階別には `map`（216条件の合成状態マップ）、`neutral`（行動0の検証）、`history`（旧9実験の診断・制御メタデータ修復・既知制御の経路検証）、`train`（固定/線形/power8 × A/B/C × 3 seed）、`report` を指定できる。`history` と `train` はローカル利用許諾の確認フラグを付ける。旧Phase 2の録音区間・共通ゲイン・保存結果を用いるので、この作業環境での再実行を想定する。実験設定は `configs/issue9.json`。

`f0` は内部名を保ち、図では外力 `$f_{e,l}$`, `$f_{e,r}$` と表記する。`requested_controls` は要求値、`physical_controls` はCase Bの強制固定後の実適用値。位置の平均と運動量は `motion_mean`, `motion_std` で区別する。訓練中の数値発散数は `training_numerical_failures` に記録する。

発声側外力を固定する対照と、停止側も残す新写像をPhase 2単独で使う場合は、旧結果を上書きしない出力先を明示する。

```bash
Deeplearning/.venv/bin/python -m Deeplearning.phase2 --target Deeplearning/sample_song/14412_0.35s.wav --config Deeplearning/configs/phase2_fixed.json --output Deeplearning/private_runs/phase2_fixed --local-permission-confirmed
Deeplearning/.venv/bin/python -m Deeplearning.phase2 --target Deeplearning/sample_song/14412_0.35s.wav --config Deeplearning/configs/phase2_power8.json --output Deeplearning/private_runs/phase2_power8 --local-permission-confirmed
```

元の `configs/phase2.json` は旧線形写像の再現用として残している。固定は診断の対照であり、外力を恒久的に固定する仕様変更ではない。新写像も模倣の成功を保証しない。これらの結果には `private_analysis.py` の旧Phase 2専用レポートをそのまま適用せず、Issue #9の保存スキーマ・実験設定を用いて比較する。

API の参照: [Gymnasium Env](https://gymnasium.farama.org/api/env/)、[SB3 custom environments](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)、[SB3 examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)。
