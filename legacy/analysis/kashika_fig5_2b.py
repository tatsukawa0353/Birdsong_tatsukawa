# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # 1. 追加
from scipy.signal import spectrogram

# 2. スタイル適用（既存の設定より前に書くのがポイント）
plt.style.use(['science', 'ieee'])

# --- ここで設定を変更できます ---
# 読み込むCSVファイル名を指定
# 例: 'simulation_output_fig5a_final.csv'
csv_filename = 'simulation-rk4_output_1(b).csv'

# スペクトログラムの解像度
nperseg = 245760
noverlap = 184320
# --------------------------------

# CSVファイルを読み込む
df = pd.read_csv(csv_filename)

time = df['time'].values
pi = df['pi'].values
total_time = time[-1]

# サンプリング周波数を計算
sampling_rate = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1.0

# 信号の末尾にパディングを追加
padding_length = nperseg 
pi_padded = np.pad(pi, (0, padding_length), 'constant')

# 【最終調整】窓関数を 'blackmanharris' に変更して、線をよりシャープにする
f, t, Sxx = spectrogram(pi_padded, fs=sampling_rate, window='blackmanharris', nperseg=nperseg, noverlap=noverlap)

# 1.結果を元のシミュレーション時間に合わせて切り取る
valid_indices = np.where(t <= total_time)[0]
# 2. 境界効果の影響が最も大きい最後の数フレームを意図的に除外する
if len(valid_indices) > 2:
    valid_indices = valid_indices[:-2]

t = t[valid_indices]
Sxx = Sxx[:, valid_indices]

# パワーを正規化し、デシベル(dB)に変換
Sxx_normalized = Sxx / np.max(Sxx) if np.max(Sxx) > 0 else Sxx
db_Sxx = 10 * np.log10(Sxx_normalized + 1e-10)

# 【最終調整】連続的なグレースケールに戻す
cmap = 'gray_r'
vmax = -40
vmin = -41

# プロット
plt.figure(figsize=(12, 12))
plt.pcolormesh(t, f, db_Sxx, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)

# --- フォントサイズの設定 ---
label_size = 46   # 軸ラベル(Frequency, Time)の大きさ
tick_size = 46 # 目盛りの数字の大きさ

for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')  # 枠の色を黒にする
        spine.set_linewidth(2.0)      # 枠の太さを設定（お好みで調整してください）

# 目盛りの設定を追加
plt.tick_params(
    axis='both', 
    which='major', 
    labelsize=tick_size, 
    colors='black',      # 目盛りの数字と線を黒にする
    width=2.0,
    length=10,           # 目盛り線の長さ（お好みで）
    direction='in',      # SciencePlots風に内向きにする
    top=False,            # 上側にも目盛りを表示
    right=False           # 右側にも目盛りを表示
)

plt.ylabel('Frequency [Hz]', fontsize=label_size)
plt.xlabel('Time [s]', fontsize=label_size)
plt.tick_params(axis='both', which='major', labelsize=tick_size)
plt.xticks([0.02, 0.04, 0.06, 0.08, 0.10])
plt.ylim(0, 10000)
plt.tight_layout()

# グラフを画像ファイルとして保存
plt.savefig('sonogram-rk4_1(b).png', bbox_inches='tight', pad_inches=0.1)
print("グラフを sonogram-rk4_1(b)_03s.png という名前で保存しました。")