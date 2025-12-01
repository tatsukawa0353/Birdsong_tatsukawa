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
    "simulation_results_2_x0=0.02_low parameters epsilon/",
    "simulation_results_2_x0=0.02/"
]
OUTPUT_3D_IMAGE = "complexity_3d_optimized_2.png"

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
    
    print(f"フォルダ読み込み: {folder}")
    for i, csv_filepath in enumerate(csv_files):
        filename = os.path.basename(csv_filepath)
        match = pattern.match(filename)
        if not match: continue

        # パラメータの丸め処理 (重複対策)
        raw_eps = float(match.group(1))
        raw_ps = float(match.group(2))
        eps = float(f"{raw_eps:.1e}")
        ps = float(f"{raw_ps:.1e}")
        
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

Z_matrix = np.zeros((len(ps_axis), len(epsilon_axis)))
res_map = {}
for _, row in df_results.iterrows():
    res_map[(row['epsilon'], row['ps'])] = row['complexity']
for i, ps in enumerate(ps_axis):
    for j, eps in enumerate(epsilon_axis):
        Z_matrix[i, j] = res_map.get((eps, ps), 0.0)


# --- ブロックサイズと位置の計算 ---
x_pos = []
y_pos = []
z_pos = []
dx = []
dy = []
dz = []

# 各軸の隙間（diff）を計算
ps_diffs = np.diff(ps_axis)
ps_diffs = np.append(ps_diffs, ps_diffs[-1])

eps_diffs = np.diff(epsilon_axis)
eps_diffs = np.append(eps_diffs, eps_diffs[-1])

for i, ps in enumerate(ps_axis):
    # Y方向（奥行き）: 少し太め(0.8)で固定
    current_dy = ps_diffs[i] * 0.8
    
    for j, eps in enumerate(epsilon_axis):
        val = Z_matrix[i, j]
        
        # --- 【修正箇所】 ---
        # 以前入れていた「上限キャップ（min_step * 2.0）」を撤廃しました。
        # 代わりに、隣のデータまでの距離の「95%」をそのままブロック幅にします。
        # これにより、右側の広い領域ではブロックが「巨大な直方体」になり、ペラペラ感が消えます。
        current_dx = eps_diffs[j] * 0.8
        # ------------------
        
        # ここは実数値(eps, ps)を使います
        x_pos.append(eps)
        y_pos.append(ps)
        z_pos.append(0)
        
        dx.append(current_dx)
        dy.append(current_dy)
        dz.append(val)

# 色の設定
colors_list = ['#ffffff', "#91CFF5", "#FF0033"] 
cmap_custom = mcolors.LinearSegmentedColormap.from_list('white_to_red', colors_list)
colors = [cmap_custom(h) for h in dz]

# --- 配列変換とマスク処理 (浮き出し削除) ---
x_pos = np.array(x_pos)
y_pos = np.array(y_pos)
z_pos = np.array(z_pos)
dx = np.array(dx)
dy = np.array(dy)
dz = np.array(dz)
colors = np.array(colors)

mask = dz > 0.001
x_pos = x_pos[mask]
y_pos = y_pos[mask]
z_pos = z_pos[mask]
dx = dx[mask]
dy = dy[mask]
dz = dz[mask]
colors = colors[mask]

# --- 【重要】描画順序の並び替え (ソート) ---
# カメラは手前(x小, y小)にあるため、奥(x大, y大)から順に描画する必要があります。
# 「X座標 + Y座標」が大きい順（降順）に並び替えます。
# ※XとYの桁数が違うため、正規化してスコア付けします
x_norm = (x_pos - x_pos.min()) / (x_pos.max() - x_pos.min())
y_norm = (y_pos - y_pos.min()) / (y_pos.max() - y_pos.min())
score = x_norm + y_norm

# スコアが大きい順（奥→手前）のインデックスを取得
sort_order = np.argsort(score)[::-1] 

# データを並び替え
x_pos = x_pos[sort_order]
y_pos = y_pos[sort_order]
z_pos = z_pos[sort_order]
dx = dx[sort_order]
dy = dy[sort_order]
dz = dz[sort_order]
colors = colors[sort_order]
# ----------------------------------------

# --- 描画 ---
print(f"描画処理中... (描画ブロック数: {len(dz)})")
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# shade=True で立体感を出す
ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True, edgecolor='k', linewidth=0.05)

# 軸の設定
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 10

ax.set_xlabel('Epsilon (ε)', fontsize=LABEL_FONTSIZE, labelpad=20)
ax.set_ylabel('Pressure (ps)', fontsize=LABEL_FONTSIZE, labelpad=20)
ax.set_zlabel('Relative Complexity', fontsize=LABEL_FONTSIZE, labelpad=10)

# 目盛り設定 (実数値座標なので、データの値を3つおきに採用)
x_ticks = epsilon_axis[::3]
ax.set_xticks(x_ticks)
ax.set_xticklabels([f"{val:.1e}" for val in x_ticks], 
                   rotation=45, fontsize=TICK_FONTSIZE)

y_ticks = ps_axis[::3]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{val:.1e}" for val in y_ticks], 
                   fontsize=TICK_FONTSIZE, rotation=-15)

ax.set_zlim(0, 1.0)
ax.view_init(elev=30, azim=-60)

plt.title('3D Complexity Map', fontsize=20)
plt.tight_layout()

print("ファイルを保存しています...")
plt.savefig(OUTPUT_3D_IMAGE, dpi=150)
print(f"完了: {OUTPUT_3D_IMAGE} に保存しました。")