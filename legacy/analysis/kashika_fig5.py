# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.signal import spectrogram

# 黒で表示するためのdB値のしきい値
# この値より強い信号が黒で表示される
threshold_db = 0
# --------------------------------

# CSVファイルを読み込む
df = pd.read_csv('simulation-rk4_output_1(a).csv')

time = df['time'].values
pi = df['pi'].values

# サンプリング周波数を計算
sampling_rate = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1.0

# ソノグラムを計算
f, t, Sxx = spectrogram(pi, fs=sampling_rate, nperseg=245760, noverlap=184320)

# パワーを正規化し、デシベル(dB)に変換
Sxx_normalized = Sxx / np.max(Sxx) if np.max(Sxx) > 0 else Sxx
db_Sxx = 10 * np.log10(Sxx_normalized + 1e-10)

# --- 【修正点】白と黒の2色だけのカスタムカラーマップを作成 ---
# カラーマップを定義: 最初の色は閾値より下(白)、次の色は閾値より上(黒)
colors = ["white", "black"]
cmap = ListedColormap(colors)

# カラーバーの境界を設定
bounds = [-80, threshold_db, 0] # -80dBからthreshold_dbまでが白、そこから0dBまでが黒
norm = plt.Normalize(vmin=-80, vmax=0)
# ----------------------------------------------------

# プロット
plt.figure(figsize=(10, 10))
# pcolormeshではvmin/vmaxの代わりにnormとboundsを使って色分けを制御
plt.pcolormesh(t, f, db_Sxx, shading='gouraud', cmap=cmap, norm=plt.Normalize(vmin=bounds[0], vmax=bounds[-1]))

plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(0, 10000)
plt.title('Spectrogram of Simulated Birdsong (pi) Fig.5(a)')

# カラーバーもカスタム設定を反映
 #cbar = plt.colorbar(ticks=[bounds[0], threshold_db, 0])
 #cbar.set_label('Intensity [dB]')

# グラフを画像ファイルとして保存
plt.savefig('sonogram-rk4_1(a).png')
print("グラフを sonogram-rk4_1(a).png という名前で保存しました。")