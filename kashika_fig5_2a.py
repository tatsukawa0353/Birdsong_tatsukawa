# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# --- ここで設定を変更できます ---
# 読み込むCSVファイル名を指定
# 例: 'simulation_output_fig5a_final.csv'
csv_filename = 'simulation-rk4_output_1(a)_03s.csv'

# スペクトログラムの解像度
nperseg = 737280#245760
noverlap = 552960#184320
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
vmax = -38
vmin = -38.002

# プロット
plt.figure(figsize=(10, 10))
plt.pcolormesh(t, f, db_Sxx, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)

plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(0, 10000)
plt.title(f'Spectrogram of Simulated Birdsong (pi) Fig.5(a)')
#plt.colorbar(label='Intensity [dB]')

# グラフを画像ファイルとして保存
plt.savefig('sonogram-rk4_1(a)_03s.png')
print("グラフを sonogram-rk4_1(a)_03s.png という名前で保存しました。")