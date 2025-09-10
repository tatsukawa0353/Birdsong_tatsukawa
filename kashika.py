# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# CSVファイルを読み込む
df = pd.read_csv('simulation_output_2(a).csv')

time = df['time'].values
pi = df['pi'].values

# サンプリング周波数を計算
sampling_rate = 1.0 / (time[1] - time[0])

# ソノグラムを計算してプロット
f, t, Sxx = spectrogram(pi, fs=sampling_rate, nperseg=2048, noverlap=1536)

plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='magma')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(0, 10000) # 論文の図に合わせて周波数範囲を設定
plt.title('Spectrogram of Simulated Birdsong (pi)')
plt.colorbar(label='Intensity [dB]')
plt.savefig('sonogram_2(a).png') 
print("グラフを sonogram_2(a).png という名前で保存しました。")