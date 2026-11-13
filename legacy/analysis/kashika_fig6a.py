# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # 1. 追加
from scipy.signal import spectrogram

# 2. スタイル適用（既存の設定より前に書くのがポイント）
plt.style.use(['science', 'ieee'])

# CSVファイルを読み込む
df = pd.read_csv('simulation-rk4_output_2(a).csv')

time = df['time'].values
pi = df['pi'].values

# サンプリング周波数を計算
sampling_rate = 1.0 / (time[1] - time[0])

# ソノグラムを計算してプロット
f, t, Sxx = spectrogram(pi, fs=sampling_rate, window='blackmanharris', nperseg=245760, noverlap=184320) #2(a)は245760　184320

# デシベル(dB)に変換
db_Sxx = 10 * np.log10(Sxx + 1e-10)

# カラースケールの調整 
cmap = 'gray_r'

# 2. 表示する色の範囲(ダイナミックレンジ)を決定
#vmax = np.max(db_Sxx)  # データ全体の最大値を取得し、これを一番濃い色(黒)とする
vmax = 80
vmin = 55       

plt.figure(figsize=(12, 12))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)

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
#plt.colorbar(label='Intensity [dB]')
plt.savefig('sonogram-rk4_2(a).png', bbox_inches='tight', pad_inches=0.1) 
print("グラフを sonogram-rk4_2(a).png という名前で保存しました。")