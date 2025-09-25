# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# CSVファイルを読み込む
df = pd.read_csv('simulation-rk4_output_2(b).csv')

time = df['time'].values
pi = df['pi'].values

# サンプリング周波数を計算
sampling_rate = 1.0 / (time[1] - time[0])

# ソノグラムを計算してプロット
f, t, Sxx = spectrogram(pi, fs=sampling_rate, nperseg=81920, noverlap=61440) #もともと2048，1536　→　40960,30720 →　8192,6144

# デシベル(dB)に変換
db_Sxx = 10 * np.log10(Sxx + 1e-10)

# カラースケールの調整 
cmap = 'gray_r'

# 2. 表示する色の範囲(ダイナミックレンジ)を決定
vmax = np.max(db_Sxx)  # データ全体の最大値を取得し、これを一番濃い色(黒)とする
vmin = vmax - 30       # 最大値から50dB下までを色の範囲とする.9/19時点30 9/22時点18

plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(0, 10000) # 論文の図に合わせて周波数範囲を設定
plt.title('Spectrogram of Simulated Birdsong (pi)')
plt.colorbar(label='Intensity [dB]')
plt.savefig('sonogram-rk4_2(b).png') 
print("グラフを sonogram-rk4_2(b).png という名前で保存しました。")