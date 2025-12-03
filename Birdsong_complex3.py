# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
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
OUTPUT_CSV_FILE = "complexity_data_weighted.csv"
OUTPUT_3D_IMAGE = "complexity_3d_weighted.png"

# 分析設定
nperseg_local = 245760 
noverlap_local = 184320
window_type = 'blackmanharris'

def calculate_weighted_spectral_entropy(csv_filepath):
    """
    重み付けスペクトル・エントロピー
    高周波成分の影響を少し弱めることで、倍音過多による複雑度上昇を抑える
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
            
        # --- 【改良】重み付け (高周波ほど値を小さく評価する) ---
        # 周波数インデックスに基づく重み (1 / (1 + log(index))) などを掛ける
        # これにより、高次倍音（インデックスが大きい）の影響力が相対的に下がる
        weights = 1.0 / (np.log1p(np.arange(len(mean_spectrum))) + 1.0)
        weighted_spectrum = mean_spectrum * weights
        
        # 正規化
        psd_norm = weighted_spectrum / np.sum(weighted_spectrum)
        
        # エントロピー計算
        ent = entropy(psd_norm, base=2)
        max_entropy = np.log2(len(psd_norm))
        return ent / max_entropy

    except Exception:
        return 0.0

# --- メイン処理 ---
print("解析開始 (Weighted Spectral Entropy)...")

results = []
pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")

for folder in INPUT_FOLDERS:
    if not os.path.exists(folder): continue
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    
    print(f"フォルダ読み込み: {folder}")
    for i, csv_filepath in enumerate(csv_files):
        filename = os.path.basename(csv_filepath)
        match = pattern.match(filename)
        if not match: continue

        raw_eps = float(match.group(1))
        raw_ps = float(match.group(2))
        eps = float(f"{raw_eps:.2e}")
        ps = float(f"{raw_ps:.2e}")
        
        # 重み付けエントロピーを使用
        raw_complexity = calculate_weighted_spectral_entropy(csv_filepath)
        
        results.append({'epsilon': eps, 'ps': ps, 'raw_complexity': raw_complexity})
        if (i+1) % 50 == 0: print(f"    Progress: {i+1} files")

df_results = pd.DataFrame(results)
df_results = df_results.drop_duplicates(subset=['epsilon', 'ps'], keep='last')

# 正規化
max_val = df_results['raw_complexity'].max()
print(f"★ Max Entropy: {max_val:.4f}")
if max_val > 0:
    df_results['normalized_complexity'] = df_results['raw_complexity'] / max_val
else:
    df_results['normalized_complexity'] = 0.0

df_results.to_csv(OUTPUT_CSV_FILE, index=False)

# --- 描画 ---
print("描画準備...")
epsilon_axis = sorted(df_results['epsilon'].unique())
ps_axis = sorted(df_results['ps'].unique())

Z_matrix = np.zeros((len(ps_axis), len(epsilon_axis)))
res_map = {}
for _, row in df_results.iterrows():
    res_map[(row['epsilon'], row['ps'])] = row['normalized_complexity']
for i, ps in enumerate(ps_axis):
    for j, eps in enumerate(epsilon_axis):
        Z_matrix[i, j] = res_map.get((eps, ps), 0.0)

# ブロック計算
x_pos = []
y_pos = []
z_pos = []
dx = []
dy = []
dz = []

ps_diffs = np.diff(ps_axis)
ps_diffs = np.append(ps_diffs, ps_diffs[-1])
eps_diffs = np.diff(epsilon_axis)
eps_diffs = np.append(eps_diffs, eps_diffs[-1])

min_step = np.min(eps_diffs[eps_diffs > 0])
max_allowed_width = min_step * 3.0

for i, ps in enumerate(ps_axis):
    current_dy = ps_diffs[i] * 0.8
    for j, eps in enumerate(epsilon_axis):
        val = Z_matrix[i, j]
        current_dx = min(eps_diffs[j] * 0.95, max_allowed_width)
        
        x_pos.append(eps)
        y_pos.append(ps)
        z_pos.append(0)
        dx.append(current_dx)
        dy.append(current_dy)
        dz.append(val)

x_pos = np.array(x_pos)
y_pos = np.array(y_pos)
z_pos = np.array(z_pos)
dx = np.array(dx)
dy = np.array(dy)
dz = np.array(dz)

mask = dz > 0.001
x_pos = x_pos[mask]
y_pos = y_pos[mask]
z_pos = z_pos[mask]
dx = dx[mask]
dy = dy[mask]
dz = dz[mask]

colors_list = ['#ffffff', '#87CEFA', '#DC143C'] 
cmap_custom = mcolors.LinearSegmentedColormap.from_list('white_to_red', colors_list)
colors = np.array([cmap_custom(h) for h in dz])

# 描画順ソート
score = x_pos + y_pos 
sort_order = np.argsort(score)[::-1] 
x_pos = x_pos[sort_order]
y_pos = y_pos[sort_order]
z_pos = z_pos[sort_order]
dx = dx[sort_order]
dy = dy[sort_order]
dz = dz[sort_order]
colors = colors[sort_order]

print("描画中...")
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)

LABEL_FONTSIZE = 14
TICK_FONTSIZE = 10
ax.set_xlabel('Epsilon (ε)', fontsize=LABEL_FONTSIZE, labelpad=20)
ax.set_ylabel('Pressure (ps)', fontsize=LABEL_FONTSIZE, labelpad=20)
ax.set_zlabel('Relative Complexity (Weighted)', fontsize=LABEL_FONTSIZE, labelpad=10)

x_ticks = epsilon_axis[::3]
ax.set_xticks(x_ticks)
ax.set_xticklabels([f"{val:.2e}" for val in x_ticks], rotation=45, fontsize=TICK_FONTSIZE)

y_ticks = ps_axis[::3]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{val:.2e}" for val in y_ticks], fontsize=TICK_FONTSIZE, rotation=-15)

ax.set_zlim(0, 1.0)
ax.view_init(elev=30, azim=-60)
plt.title('3D Complexity Map (Weighted Spectral Entropy)', fontsize=20)
plt.tight_layout()

print("保存中...")
plt.savefig(OUTPUT_3D_IMAGE, dpi=150)
print(f"完了: {OUTPUT_3D_IMAGE}")