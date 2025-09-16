# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# CSVファイルを読み込む
df = pd.read_csv('simulation_output_2(a).csv')

time = df['time'].values
x_left = df['x_left'].values
x_right = df['x_right'].values

# グラフプロット(並べて）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

#上グラフ(左音源)
ax1.plot(time, x_left, color='royalblue')
ax1.set_title('Left Source Waveform')
ax1.set_ylabel('Displacement [cm]')
ax1.grid(True, linestyle='--', alpha=0.6)

#下グラフ(右音源)
ax2.plot(time, x_right, color='seagreen')
ax2.set_title('Right Source Waveform')
ax2.set_ylabel('Displacement [cm]')
ax2.set_xlabel('Time [sec]')
ax2.grid(True, linestyle='--', alpha=0.6)

# レイアウトを整える
plt.tight_layout()

# グラフを画像ファイルとして保存
output_filename = 'waveform_2(a).png'
plt.savefig(output_filename)

print(f"振動波形グラフを {output_filename} という名前で保存しました。")