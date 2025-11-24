# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import matplotlib
matplotlib.use('Agg') # ウィンドウを表示せず保存に専念

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import spectrogram
from scipy.stats import entropy
import glob
import os
import re

# --- 【重要】統合したいフォルダのリスト ---
INPUT_FOLDERS = [
    "simulation_results_1_x0=0.02_low parameters epsilon/", # 低ε範囲
    "simulation_results_1_x0=0.02/"                         # 高ε範囲
]
# ----------------------------------------

OUTPUT_3D_IMAGE = "complexity_3d_combined_1.png"

# 分析設定
nperseg_local = 245760 
noverlap_local = 184320
window_type = 'blackmanharris'

def calculate_spectral_entropy_full(csv_filepath):
    """ CSVファイルを読み込み、全データを使ってエントロピーを計算する """
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

    except Exception as e:
        print(f"Error in {csv_filepath}: {e}")
        return 0.0

# --- メイン処理 ---
print("複数のフォルダからデータを統合して解析します...")

results = []
pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")

# 1. フォルダをループして全データを収集
for folder in INPUT_FOLDERS:
    print(f"\nフォルダ読み込み中: {folder}")
    if not os.path.exists(folder):
        print(f"  [警告] フォルダが見つかりません: {folder}")
        continue
        
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    print(f"  -> {len(csv_files)} 個のファイルが見つかりました。計算を開始します...")
    
    for i, csv_filepath in enumerate(csv_files):
        filename = os.path.basename(csv_filepath)
        match = pattern.match(filename)
        if not match: continue

        eps = float(match.group(1))
        ps = float(match.group(2))
        
        complexity = calculate_spectral_entropy_full(csv_filepath)
        results.append({'epsilon': eps, 'ps': ps, 'complexity': complexity})
        
        if (i+1) % 50 == 0:
            print(f"    Progress: {i+1}/{len(csv_files)}")

# データフレーム化
df_results = pd.DataFrame(results)

# 重複データの削除 (もし両方のフォルダに同じ条件のファイルがあった場合、後勝ちにする)
df_results = df_results.drop_duplicates(subset=['epsilon', 'ps'], keep='last')

print(f"\n総データ数: {len(df_results)} 件")

# 2. 正規化 (最大値を1.0にする)
max_val = df_results['complexity'].max()
if max_val > 0:
    print(f"全体での最大エントロピー: {max_val:.4f} -> これを 1.0 に変換します")
    df_results['complexity'] = df_results['complexity'] / max_val

print("3Dグラフの描画準備に入ります...")

# --- 3Dグラフデータの準備 ---
# ここで sorted() を使うことで、異なるフォルダのデータがきれいに結合・整列されます
epsilon_axis = sorted(df_results['epsilon'].unique())
ps_axis = sorted(df_results['ps'].unique())

# グリッド作成 (インデックス座標)
x_indices = np.arange(len(epsilon_axis))
y_indices = np.arange(len(ps_axis))
X_idx, Y_idx = np.meshgrid(x_indices, y_indices)

# Zマトリックスを埋める
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

# 色の設定
cmap = matplotlib.colormaps['coolwarm'] 
colors = [cmap(h) for h in dz] # h is already 0.0-1.0

# --- 描画 ---
print("描画処理中...")
fig = plt.figure(figsize=(16, 12)) # 横幅を少し広げました
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)

# 軸ラベル
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 10

ax.set_xlabel('Epsilon (ε)', fontsize=LABEL_FONTSIZE, labelpad=15)
ax.set_ylabel('Pressure (ps)', fontsize=LABEL_FONTSIZE, labelpad=15)
ax.set_zlabel('Relative Complexity', fontsize=LABEL_FONTSIZE, labelpad=10)

# --- X軸の目盛り設定 (ここがポイント) ---
# データ数が多くなるので、ラベルが重ならないよう適切に間引く
total_x_points = len(epsilon_axis)
x_step = max(1, total_x_points // 15) # ラベル数を15個くらいに抑える

ax.set_xticks(x_indices[::x_step])
ax.set_xticklabels([f"{epsilon_axis[i]:.1e}" for i in range(0, total_x_points, x_step)], 
                   rotation=45, fontsize=TICK_FONTSIZE)

# Y軸の目盛り設定
total_y_points = len(ps_axis)
y_step = max(1, total_y_points // 10)

ax.set_yticks(y_indices[::y_step])
ax.set_yticklabels([f"{ps_axis[i]:.1e}" for i in range(0, total_y_points, y_step)], 
                   fontsize=TICK_FONTSIZE, rotation=-15)

ax.set_zlim(0, 1.0)
ax.view_init(elev=30, azim=-60)

plt.title('Combined 3D Complexity Map', fontsize=20)
plt.tight_layout()

print("ファイルを保存しています...")
plt.savefig(OUTPUT_3D_IMAGE, dpi=150)
print(f"完了: 統合された3Dヒストグラムを {OUTPUT_3D_IMAGE} に保存しました。")