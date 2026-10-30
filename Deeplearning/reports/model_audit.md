# 既存モデルの調査・再利用仕様

調査日: 2026-10-30（JST）。対象: [Issue #1](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/1)、[Issue #4](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/4)。既存ファイルは変更していない。調査時の SHA-256 と基準コミットは `results/environment.json` に記録した。

## 数値モデル

鳴管は1器官で、左右に独立に制御し得る2つの音源がある。各側の状態は位置 `x_j` と速度 `y_j`。既存コードで実際に積分される式は次のとおり。

```text
dx_j/dt = y_j
dy_j/dt = -epsilon_j*x_j - b*y_j - c*x_j^2*y_j - f0_j
          + ps + (D*y_j-A)*(ps-pi_tilde)
pi_tilde = Σ_j {alpha_j*(x_j-tau*y_j)+beta_j*y_j} - gamma*pi_delayed
pi_next  = pi_tilde_next - Σ_j beta_j*tau*(dy_j/dt)_next
```

`f0` は研究上の外力項 `f_e` に対応し、音響的な基本周波数 F0 と区別する。上式の `A` はモデル係数であり、比較条件 A とは別の意味。

| 項目 | 既存の値・扱い |
|---|---|
| `dt` | `1e-7 s`、変更していない |
| `L`, `c_sound` | `2 cm`, `3.5e4 cm/s` |
| 遅延 | `2L/c_sound = 1.142857142857e-4 s` |
| 遅延バッファ | `int(T_delay/dt)=1142` 点、実効遅延 `114.2 µs` |
| 初期状態 | 左右とも `x=y=0`、履歴ゼロ、時刻ゼロ |
| `b`, `c` | `2e4`, `2e8` |
| `f0` | 基準 `4.9e4` |
| `tau`, `A`, `D`, `gamma` | `1e-4`, `1.0`, `0.01`, `0.9` |
| `alpha`, `beta` | 基準 `1.2e6`, `1.5e3` |
| `x0`, `h`, `M` | 初期化値 `0.04`, `0.1`, `5e-3`。この実装の微分式からは参照されない |
| 単位系 | 既存数値単位系を保存。出力圧力と WAV 振幅の SI 対応は校正されていない |

RK4 の4段では同じ `pi_delayed` を用い、中間状態ごとに `pi_tilde` を再計算する。最終状態から加速度と出力圧力を再計算して履歴へ格納する。積分中に遅延補間しない。リングバッファの `current_pos` が読み出す最古値であり、その位置へ新しい出力を上書きする。`reset(history=...)` は最古→最新の順を受け取り、カーソルをゼロに戻す。

この離散遅延の実装まで保存することを回帰基準にした。論文の連続方程式を独立に検証したものではない。精度確認なしの刻み拡大、初期過渡の削除、グリッド探索の移植は行っていない。

## 実装差と音源条件

| 既存ファイル | 実際の設定・注意点 |
|---|---|
| `BirdsongModel_10*10.cpp` | 右 `alpha=beta=0`。右の力学状態は存在し、左と同様に振動し得るが出力への寄与は除外される。条件 A の基準 |
| `BirdsongModel_10*10_2.cpp` | 左右とも寄与し、左右別々の `epsilon` を設定。右 `f0` 上書きはコメントアウトされている。条件 C に相当する基準 |
| `BirdsongModel_10*10_s(t).cpp` | 右 `f0=0.1e7`。`s_l`, `s_r` を出力。ただし現行 `BirdsongModel_2.h` には `saveData(double,double)` 宣言がなく、そのままでは組み合わせてコンパイルできない |
| `main_10*10.cpp` | 名称と異なり26×16=416条件。`epsilon=2.5e7..5e7`, `ps=2.5e6..4e6`、各0.12秒 |
| `main_10*10_2.cpp` | 左右別 `epsilon` 用のドライバ。掃引処理は新環境に持ち込んでいない |

新 API の条件 B は右の寄与係数を残し `f0_r=1e7` を固定する仮説条件とした。**右の振動抑制を数式上で保証する条件ではない**。保存した制御値と追加の運動量評価から確認する必要がある。左右両方の発声を「鳴管が2器官ある」とは表現しない。

従来の CSV 列は `time,pi,x_left,y_left,x_right,y_right`。時刻は積分後の `t+dt`、`saveData()` の初回のみ `t=0` の行となる。`s(t)` 版は `time,pi,s_l,s_r,x_left,y_left,x_right,y_right`。保存は既定6桁のため、新旧比較の許容誤差を `rtol=5.1e-6, atol=5.1e-7` とした。

## Python 接続

接続は C ABI + `ctypes`。pybind11 のヘッダ依存や毎ステップのファイル I/O を避け、C++ 側で指定期間をまとめて積分する構成を採用した。pybind11 との実測速度比較は未実施。`python -m Deeplearning.model` でコンパイルし、ライブラリは原子的に置き換える。`-ffast-math` は使っていない。

```python
from Deeplearning.model import BirdModel

with BirdModel("A") as model:
    model.reset()  # 状態・履歴・時刻・制御値
    model.set_controls([3.5e7, 3.5e7, 3e6, 4.9e4, 4.9e4])
    pressure = model.advance(0.01)  # 内部では100,000 RK4ステップ
    state = model.state  # time, pi, x_l, y_l, x_r, y_r
```

`advance` を分割しても履歴を保持し、分割なしの結果とビット単位で一致する。Python では制御値の有限性・範囲・形状、時間の積分刻みとの整合を検証する。発散後はリセットまで再利用できない。C++ の例外は ABI 境界を越えない。

Issue #9で `advance(..., return_moments=True)` を追加し、全積分点の位置・速度の平均と標準偏差を取得できるようにした。これは運動の診断でありRK4の式・dt・遅延を変更しない。要求値と実適用値の変換は `effective_controls()` に統一し、Case Bの右外力1e7を保存値と図へも反映する。既存 `return_rms=True` APIも維持した。

Phase 1 の可動値は `epsilon_l`, `ps` の2個。API 自体は左右 `epsilon`、共通 `ps`、左右 `f0` の5個を更新できる。範囲は `epsilon=[2.5e7,5e7]`, `ps=[2.5e6,4e6]`, `f0=[4.9e4,1e7]`。範囲の大部分が発声しやすいと保証するものではなく、特に `f0` は音の停止を誘発し得る。

制御は区間内一定、左右の更新は同時。別々の自由度による非対称性は許すが、交互発声を強制する制約は実装していない。観測は時刻・直前行動・4状態の正規化値であり、遅延履歴全体を含まない部分観測である。

## STFT と図の再利用

既存の `kashika_10*10.py`、`kashika_10*10_2.py` は Blackman–Harris 窓、`nperseg=245760`, `noverlap=184320`。10 MHz に対し窓24.576 ms、ホップ6.144 ms、重複75%。末尾に窓長のゼロを加え、時刻で切り詰めて末尾2フレームを除く。PSD を各図の最大値で正規化し `10log10(...+1e-10)` とする。

描画は `science/ieee`、12×12 inch、`gray_r`、`vmin=-38.002`, `vmax=-38`、ラベル/目盛46 pt、枠線2、内向き目盛長10、周波数0–10000 Hz、PNG、`bbox_inches='tight', pad_inches=0.1`。`kashika_waveform10*10.py` の波形目盛は38 pt、ゼロ線0.8、枠線2。新しい図では0.35秒へ横軸を合わせる。実行環境で LaTeX を必須にしないため `no-latex` を追加した。

基準図 `results/reference/legacy_spectrogram.png` は未変更の既存C++による単一例の10 MHz波形を描画する。当初の比較図には `[-80,0] dB` を用いていたが、2026-10-30のユーザー指定により、描画をすべて `gray_r`、`vmin=-50.0`、`vmax=-20.0` に統一した。元コードのパディングと末尾フレームの切り捨ては描画に引き継がず、窓全体が音声内に収まるSTFTフレームだけを用い、横軸を最初・最後の有効フレーム中心で切る。0秒・音声終端は表示しない。実録音の従来手順に合わせ、保存音声も描画時に10 MHzへリサンプリングし、nperseg=245760・noverlap=184320をそのまま適用する。0.35秒音声の横軸は0.012288–0.331776秒となる。表示用の個別最大値正規化は報酬には用いない。

新しい報酬用 STFT は秒単位で窓とホップを決定し、中央フレームとゼロパディングで目標・生成音を同じ時刻にそろえる。従来の境界切り捨てとは区別する。鳴管出力は16次 Butterworth 15 kHz ローパスの状態を保って10 MHz→50 kHzへ200分の1に間引く。これは出力処理であり積分刻みの変更ではない。合成目標と生成音のフィルタは同じで、実録音との声道・位相・群遅延の差は未同定である。

## 基準再現

`python -m Deeplearning.reproduce` が既存の `BirdsongModel_10*10.cpp` を変更せず、単一固定条件 `epsilon=3.5e7, ps=3e6` の0.12秒を実行する。CSV は一時領域だけで扱い、比較波形・音声・図と誤差を `results/reference/` に保存する。0.35秒=3,500,000積分ステップの A/B/C 各3回の速度と有限性は `benchmark.json` に記録する。
