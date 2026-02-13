# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

#グラフで表示する時間の範囲指定
time_start = 0.0  # 開始時間
time_end = 0.05  # 終了時間

# CSVファイルを読み込む
df = pd.read_csv('simulation_output_test1.csv')

time = df['time'].values
x_left = df['x_left'].values
x_right = df['x_right'].values

# グラフプロット(並べて）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# --- フォントサイズの設定 ---
label_size = 42   # 軸ラベル(Frequency, Time)の大きさ
tick_size = 36   # 目盛りの数字の大きさ

#上グラフ(左音源)
ax1.plot(time, x_left, color='royalblue')
#ax1.set_title('Left Source Waveform')
ax1.set_ylabel('$x_l$ [cm]', fontsize=label_size)
ax2.set_xlabel('Time [s]', fontsize=label_size)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.axhline(0, color='black', linewidth=0.8)
ax1.tick_params(axis='y', labelsize=tick_size)

#下グラフ(右音源)
ax2.plot(time, x_right, color='seagreen')
#ax2.set_title('Right Source Waveform')
ax2.set_ylabel('$x_r$ [cm]', fontsize=label_size)
ax2.set_xlabel('Time [s]', fontsize=label_size)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.axhline(0, color='black', linewidth=0.8)

# --- X軸の表示範囲を指定 ---
ax2.set_xlim([time_start, time_end])
ax2.set_xticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
ax2.tick_params(axis='both', which='major', labelsize=tick_size)

# レイアウトを整える
plt.tight_layout()

# グラフを画像ファイルとして保存
output_filename = 'waveform_test1.png'
plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)

print(f"振動波形グラフを {output_filename} という名前で保存しました。")