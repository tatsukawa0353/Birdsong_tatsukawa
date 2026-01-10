# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, resample
from scipy.io import wavfile

# ==========================================
#      設定セクション
# ==========================================

# 1. ファイル設定
wav_filename = 'Wsst_1.wav' # 拡張子は実際のファイルに合わせてください(.wav等)

# 2. シミュレーション条件への合わせ込み
target_fs = 1e7  # 10^7 Hz (10 MHz)
nperseg = 245760
noverlap = 184320

# 3. プロットの見た目設定（フォントサイズ等）
fig_size = (10, 10)       # 画像の縦横サイズ (インチ)
label_fontsize = 18      # 軸ラベル（Time, Frequency）の大きさ
title_fontsize = 20      # タイトルの文字の大きさ
tick_fontsize = 14       # 目盛りの数字の大きさ
#font_family = 'Arial'    # フォントの種類（'Times New Roman', 'Arial', 'Meiryo'など）

# 4. カラーマップ設定
cmap = 'gray_r'
vmax = -35
vmin = -38.002
# ==========================================

# フォント設定を適用
#plt.rcParams['font.family'] = font_family
plt.rcParams['font.size'] = tick_fontsize # 全体の基準サイズ

# --- データ処理（前回と同じ） ---

# WAVファイルを読み込む
try:
    original_fs, data = wavfile.read(wav_filename)
except FileNotFoundError:
    print(f"エラー: ファイル '{wav_filename}' が見つかりません。")
    exit()

# ステレオ（2ch）の場合はモノラルに変換
if len(data.shape) == 2:
    data = data.mean(axis=1)

data = data.astype(float)

# サンプリングレートの調整（アップサンプリング）
if original_fs != target_fs:
    print(f"サンプリング周波数を変換します: {original_fs}Hz -> {target_fs}Hz")
    num_samples_target = int(len(data) * target_fs / original_fs)
    pi = resample(data, num_samples_target)
    sampling_rate = target_fs
else:
    pi = data
    sampling_rate = original_fs

# 時間軸を作成
time = np.arange(0, len(pi)) / sampling_rate
total_time = time[-1]

# 信号の末尾にパディング
padding_length = nperseg 
pi_padded = np.pad(pi, (0, padding_length), 'constant')

# スペクトログラム計算
f, t, Sxx = spectrogram(pi_padded, fs=sampling_rate, window='blackmanharris', nperseg=nperseg, noverlap=noverlap)

# 後処理
valid_indices = np.where(t <= total_time)[0]
if len(valid_indices) > 2:
    valid_indices = valid_indices[:-2]

t = t[valid_indices]
Sxx = Sxx[:, valid_indices]

# dB変換
Sxx_normalized = Sxx / np.max(Sxx) if np.max(Sxx) > 0 else Sxx
db_Sxx = 10 * np.log10(Sxx_normalized + 1e-10)


# --- プロット処理（設定を反映） ---

plt.figure(figsize=fig_size)

# スペクトログラム描画
plt.pcolormesh(t, f, db_Sxx, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)

# 軸ラベルの設定
plt.ylabel('Frequency [Hz]', fontsize=label_fontsize)
plt.xlabel('Time [sec]', fontsize=label_fontsize)

# タイトルの設定
plt.title(f'Spectrogram', fontsize=title_fontsize)

# 軸目盛りの設定
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
# 必要であれば目盛りを内向きにする（論文でよく使われます）
plt.tick_params(direction='in') 

plt.ylim(0, 10000)

# 保存
output_filename = 'sonogram-wav-Wsst_1.png'
plt.savefig(output_filename, bbox_inches='tight', dpi=300) # dpi=300で高画質保存
print(f"グラフを {output_filename} として保存しました。")