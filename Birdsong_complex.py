# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import spectrogram
from scipy.stats import entropy
import glob
import os
import re

# --- 統合したいフォルダ ---
INPUT_FOLDERS = [
    "simulation_results_1_x0=0.02_low parameters epsilon/",
    "simulation_results_1_x0=0.02/"
]
OUTPUT_3D_IMAGE = "complexity_3d_optimized_1.png"

# 分析設定
nperseg_local = 245760 
noverlap_local = 184320
window_type = 'blackmanharris'

def calculate_spectral_entropy_full(csv_filepath):
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
        return ent / max_entropy

    except Exception:
        return 0.0

# --- メイン処理 ---
print("複数のフォルダからデータを統合して解析します...")

results = []
pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")

for folder in INPUT_FOLDERS:
    if not os.path.exists(folder): continue
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    for i, csv_filepath in enumerate(csv_files):
        filename = os.path.basename(csv_filepath)
        match = pattern.match(filename)
        if not match: continue
        eps = float(match.group(1))
        ps = float(match.group(2))
        complexity = calculate_spectral_entropy_full(csv_filepath)
        results.append({'epsilon': eps, 'ps': ps, 'complexity': complexity})
        if (i+1) % 50 == 0: print(f"    Progress: {i+1} files")

df_results = pd.DataFrame(results)
df_results = df_results.drop_duplicates(subset=['epsilon', 'ps'], keep='last')

# 正規化
max_val = df_results['complexity'].max()
if max_val > 0:
    df_results['complexity'] = df_results['complexity'] / max_val

print("3Dグラフのデータ準備中...")

epsilon_axis = sorted(df_results['epsilon'].unique())
ps_axis = sorted(df_results['ps'].unique())

# Zマトリックス
Z_matrix = np.zeros((len(ps_axis), len(epsilon_axis)))
res_map = {}
for _, row in df_results.iterrows():
    res_map[(row['epsilon'], row['ps'])] = row['complexity']
for i, ps in enumerate(ps_axis):
    for j, eps in enumerate(epsilon_axis):
        Z_matrix[i, j] = res_map.get((eps, ps), 0.0)


# --- 【重要】ブロックサイズと位置の計算 ---
x_pos = []
y_pos = []
z_pos = []
dx = []
dy = []
dz = []

# Y軸 (Pressure) の幅計算:
# 「最小間隔の80%」で固定してしまうと薄すぎるので、
# ここも「次の点までの距離の 80%」にして、間隔が広いところは太くします。
ps_diffs = np.diff(ps_axis)
# 最後の一つは前の間隔と同じにする
ps_diffs = np.append(ps_diffs, ps_diffs[-1])


# X軸 (Epsilon) の幅計算:
# ここも「次の点までの距離」をベースにします
eps_diffs = np.diff(epsilon_axis)
eps_diffs = np.append(eps_diffs, eps_diffs[-1])


for i, ps in enumerate(ps_axis):
    # Y方向の太さ: 次の点までの距離の 60% (少し隙間を空ける)
    current_dy = ps_diffs[i] * 0.6
    
    for j, eps in enumerate(epsilon_axis):
        val = Z_matrix[i, j]
        
        # X方向の太さ: 次の点までの距離の 90% (ギリギリまで詰める)
        current_dx = eps_diffs[j] * 0.9
        
        x_pos.append(eps)
        y_pos.append(ps)
        z_pos.append(0)
        
        dx.append(current_dx)
        dy.append(current_dy)
        dz.append(val)

# 色の設定
cmap = matplotlib.colormaps['coolwarm'] 
colors = [cmap(h) for h in dz]

# --- 描画 ---
print("描画処理中...")
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)

# --- 軸の設定 ---
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 10

ax.set_xlabel('Epsilon (ε)', fontsize=LABEL_FONTSIZE, labelpad=20)
ax.set_ylabel('Pressure (ps)', fontsize=LABEL_FONTSIZE, labelpad=20)
ax.set_zlabel('Relative Complexity', fontsize=LABEL_FONTSIZE, labelpad=10)

# 目盛り設定 (3つおき)
x_ticks = epsilon_axis[::3]
ax.set_xticks(x_ticks)
ax.set_xticklabels([f"{val:.1e}" for val in x_ticks], 
                   rotation=45, fontsize=TICK_FONTSIZE)

y_ticks = ps_axis[::3]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{val:.1e}" for val in y_ticks], 
                   fontsize=TICK_FONTSIZE, rotation=-15)

ax.set_zlim(0, 1.0)
ax.view_init(elev=40, azim=-60)

plt.title('3D Complexity Map (Dynamic Size)', fontsize=20)
plt.tight_layout()

print("ファイルを保存しています...")
plt.savefig(OUTPUT_3D_IMAGE, dpi=150)
print(f"完了: {OUTPUT_3D_IMAGE} に保存しました。")