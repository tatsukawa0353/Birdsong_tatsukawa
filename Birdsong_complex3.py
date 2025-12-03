# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import glob
import os
import sys
import re

# --- 統合したいフォルダ ---
INPUT_FOLDERS = [
    "simulation_results_1_x0=0.02_low parameters epsilon/",
    "simulation_results_1_x0=0.02/"
]
OUTPUT_CSV_FILE = "complexity_data_pe_1.csv"
OUTPUT_3D_IMAGE = "complexity_3d_pe_1.png"

# --- 順列エントロピーの設定（チューニング） ---
PE_ORDER = 4 
# 【重要】遅延時間を増やす (隣り合うデータが近すぎて変化がないのを防ぐ)
# サンプリング周波数が高い場合、ここを10〜100くらいにすると模様が見えてくる
PE_DELAY = 2000

def calculate_permutation_entropy(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty or 'pi' not in df: return 0.0
        
        x = df['pi'].values
        N = len(x)
        if N < 100: return 0.0

        m = PE_ORDER
        tau = PE_DELAY
        
        n_vectors = N - (m - 1) * tau
        if n_vectors <= 0: return 0.0
        
        # 埋め込み行列の作成
        matrix = np.empty((n_vectors, m))
        for i in range(m):
            matrix[:, i] = x[i*tau : i*tau + n_vectors]
            
        # 各行の順位パターンを取得
        # kind='stable' を指定して、同じ値の場合の順序を安定させる
        orders = np.argsort(matrix, axis=1, kind='stable')
        
        # パターンのユニークカウント
        dt = np.dtype((np.void, orders.dtype.itemsize * orders.shape[1]))
        b = np.ascontiguousarray(orders).view(dt)
        _, counts = np.unique(b, return_counts=True)
        
        # 確率分布
        p = counts / n_vectors
        p = p[p > 0]
        
        # エントロピー
        pe = -np.sum(p * np.log2(p))
        
        # 正規化
        max_entropy = np.log2(np.math.factorial(m))
        normalized_pe = pe / max_entropy
        
        return normalized_pe

    except Exception as e:
        print(f"Error in calculation: {e}")
        return 0.0

# --- メイン処理 ---
print(f"解析開始 (PE_ORDER={PE_ORDER}, PE_DELAY={PE_DELAY})...")

results = []
pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")

for folder in INPUT_FOLDERS:
    if not os.path.exists(folder): continue
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    
    print(f"フォルダ読み込み: {folder} ({len(csv_files)} files)")
    for i, csv_filepath in enumerate(csv_files):
        filename = os.path.basename(csv_filepath)
        match = pattern.match(filename)
        if not match: continue

        raw_eps = float(match.group(1))
        raw_ps = float(match.group(2))
        eps = float(f"{raw_eps:.1e}")
        ps = float(f"{raw_ps:.1e}")
        
        raw_complexity = calculate_permutation_entropy(csv_filepath)
        
        # デバッグ: 最初の数個だけ値を表示して確認
        if i < 3:
            print(f"  Debug: {filename} -> {raw_complexity:.4f}")

        results.append({
            'epsilon': eps, 
            'ps': ps, 
            'raw_complexity': raw_complexity
        })
        
        if (i+1) % 50 == 0: print(f"    Progress: {i+1} files")

df_results = pd.DataFrame(results)

if df_results.empty:
    print("【エラー】有効なデータがありません。")
    sys.exit()

df_results = df_results.drop_duplicates(subset=['epsilon', 'ps'], keep='last')

# 正規化
max_val = df_results['raw_complexity'].max()
print(f"★ 計算された最大エントロピー値: {max_val}")

if max_val > 0.000001:
    df_results['normalized_complexity'] = df_results['raw_complexity'] / max_val
else:
    print("【警告】すべてのファイルの複雑度が 0.0 です。PE_DELAY をもっと大きくしてください。")
    df_results['normalized_complexity'] = 0.0

# CSV保存
df_results = df_results.sort_values(by=['epsilon', 'ps'])
df_results.to_csv(OUTPUT_CSV_FILE, index=False)

# ---------------------------------------------------------
# グラフ描画
# ---------------------------------------------------------
print("3Dグラフのデータ準備中...")

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

for i, ps in enumerate(ps_axis):
    current_dy = ps_diffs[i] * 0.8
    for j, eps in enumerate(epsilon_axis):
        val = Z_matrix[i, j]
        current_dx = eps_diffs[j] * 0.95
        
        x_pos.append(eps)
        y_pos.append(ps)
        z_pos.append(0)
        dx.append(current_dx)
        dy.append(current_dy)
        dz.append(val)

# 配列変換
x_pos = np.array(x_pos)
y_pos = np.array(y_pos)
z_pos = np.array(z_pos)
dx = np.array(dx)
dy = np.array(dy)
dz = np.array(dz)

# マスク処理（少し緩める）
mask = dz > 0.000001
if np.sum(mask) == 0:
    print("【警告】描画データがありません。フィルタを解除します。")
    mask = np.ones(len(dz), dtype=bool)

x_pos = x_pos[mask]
y_pos = y_pos[mask]
z_pos = z_pos[mask]
dx = dx[mask]
dy = dy[mask]
dz = dz[mask]

# 色設定
colors_list = ['#ffffff', '#87CEFA', '#DC143C'] 
cmap_custom = mcolors.LinearSegmentedColormap.from_list('white_to_red', colors_list)
colors = np.array([cmap_custom(h) for h in dz])

# ソート
score = x_pos + y_pos 
sort_order = np.argsort(score)[::-1] 
x_pos = x_pos[sort_order]
y_pos = y_pos[sort_order]
z_pos = z_pos[sort_order]
dx = dx[sort_order]
dy = dy[sort_order]
dz = dz[sort_order]
colors = colors[sort_order]

# 描画
print(f"描画処理中... (描画ブロック数: {len(dz)})")
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)

LABEL_FONTSIZE = 14
TICK_FONTSIZE = 10
ax.set_xlabel('Epsilon (ε)', fontsize=LABEL_FONTSIZE, labelpad=20)
ax.set_ylabel('Pressure (ps)', fontsize=LABEL_FONTSIZE, labelpad=20)
ax.set_zlabel('Relative Complexity (PE)', fontsize=LABEL_FONTSIZE, labelpad=10)

x_ticks = epsilon_axis[::3]
ax.set_xticks(x_ticks)
ax.set_xticklabels([f"{val:.1e}" for val in x_ticks], rotation=45, fontsize=TICK_FONTSIZE)

y_ticks = ps_axis[::3]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{val:.1e}" for val in y_ticks], fontsize=TICK_FONTSIZE, rotation=-15)

ax.set_zlim(0, 1.0)
ax.view_init(elev=30, azim=-60)
plt.title(f'3D Complexity Map (PE delay={PE_DELAY})', fontsize=20)
plt.tight_layout()

print("ファイルを保存しています...")
plt.savefig(OUTPUT_3D_IMAGE, dpi=150)
print(f"完了: {OUTPUT_3D_IMAGE} に保存しました。")