# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import matplotlib
# 【重要】ウィンドウを表示せず、ファイル保存に専念する設定
# これをpyplotをインポートする前に行うことで、フリーズを回避します
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm # 不要になったので削除
from scipy.signal import spectrogram
from scipy.stats import entropy
import glob
import os
import re

# --- 設定 ---
INPUT_FOLDER = "simulation_results_1_x0=0.02_low parameters epsilon/"  # フォルダ名を確認
OUTPUT_3D_IMAGE = "complexity_3d_histogram_1_ep.png"

# kashika.py と同じ分析設定
nperseg_local = 245760 
noverlap_local = 184320
window_type = 'blackmanharris'

def calculate_spectral_entropy_full(csv_filepath):
    """
    CSVファイルを読み込み、全データを使ってエントロピーを計算する
    """
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty or 'pi' not in df or len(df) < nperseg_local:
            return 0.0

        sampling_rate = 1.0 / (df['time'].values[1] - df['time'].values[0])
        
        pi = df['pi'].values 
        
        if len(pi) < nperseg_local:
            return 0.0

        f, t, Sxx = spectrogram(pi, fs=sampling_rate, window=window_type, nperseg=nperseg_local, noverlap=noverlap_local)
        mean_spectrum = np.mean(Sxx, axis=1)
        
        if np.sum(mean_spectrum) == 0:
            return 0.0
        psd_norm = mean_spectrum / np.sum(mean_spectrum)
        
        ent = entropy(psd_norm, base=2)
        max_entropy = np.log2(len(psd_norm))
        normalized_entropy = ent / max_entropy
        
        return normalized_entropy

    except Exception as e:
        print(f"Error in {csv_filepath}: {e}")
        return 0.0

# --- メイン処理 ---
print(f"'{INPUT_FOLDER}' の解析を開始します（全データ使用・3D化）...")

csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
results = []
pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")

for i, csv_filepath in enumerate(csv_files):
    filename = os.path.basename(csv_filepath)
    match = pattern.match(filename)
    if not match: continue

    eps = float(match.group(1))
    ps = float(match.group(2))
    
    complexity = calculate_spectral_entropy_full(csv_filepath)
    results.append({'epsilon': eps, 'ps': ps, 'complexity': complexity})
    
    # 進捗表示 (頻度を減らして高速化)
    if (i+1) % 20 == 0:
        print(f"Progress: {i+1}/{len(csv_files)}")

df_results = pd.DataFrame(results)
print("計算完了。3Dグラフの描画準備に入ります...")

# --- 3Dグラフのデータの準備 ---
epsilon_axis = sorted(df_results['epsilon'].unique())
ps_axis = sorted(df_results['ps'].unique())

x_indices = np.arange(len(epsilon_axis))
y_indices = np.arange(len(ps_axis))
X_idx, Y_idx = np.meshgrid(x_indices, y_indices)

Z_matrix = np.zeros((len(ps_axis), len(epsilon_axis)))
res_map = {}
for _, row in df_results.iterrows():
    res_map[(row['epsilon'], row['ps'])] = row['complexity']

for i, ps in enumerate(ps_axis):
    for j, eps in enumerate(epsilon_axis):
        Z_matrix[i, j] = res_map.get((eps, ps), 0.0)

# フラット化
x_pos = X_idx.flatten()
y_pos = Y_idx.flatten()
z_pos = np.zeros_like(x_pos)
dx = 0.8 * np.ones_like(z_pos)
dy = 0.8 * np.ones_like(z_pos)
dz = Z_matrix.flatten()

# --- 色の設定 (修正済み) ---
# 【修正】警告が出ない新しい書き方に変更
cmap = matplotlib.colormaps['coolwarm'] 
max_height = 1.0
colors = [cmap(h / max_height) for h in dz]

# --- 描画 ---
print("描画処理中...")
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)

# 軸ラベル
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 10

ax.set_xlabel('Epsilon (ε)', fontsize=LABEL_FONTSIZE, labelpad=15)
ax.set_ylabel('Pressure (ps)', fontsize=LABEL_FONTSIZE, labelpad=15)
ax.set_zlabel('Complexity (Spectral Entropy)', fontsize=LABEL_FONTSIZE, labelpad=10)

# 目盛り設定
x_step = max(1, len(epsilon_axis) // 10)
ax.set_xticks(x_indices[::x_step])
ax.set_xticklabels([f"{epsilon_axis[i]:.1e}" for i in range(0, len(epsilon_axis), x_step)], 
                   rotation=45, fontsize=TICK_FONTSIZE)

y_step = max(1, len(ps_axis) // 10)
ax.set_yticks(y_indices[::y_step])
ax.set_yticklabels([f"{ps_axis[i]:.1e}" for i in range(0, len(ps_axis), y_step)], 
                   fontsize=TICK_FONTSIZE, rotation=-15)

ax.set_zlim(0, 1.0)
ax.view_init(elev=30, azim=-60)

plt.title('3D Complexity Map (Full Data Analysis)', fontsize=20)
plt.tight_layout()

print("ファイルを保存しています...")
plt.savefig(OUTPUT_3D_IMAGE, dpi=150)
print(f"完了: 3Dヒストグラムを {OUTPUT_3D_IMAGE} に保存しました。")