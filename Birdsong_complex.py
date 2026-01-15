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
    "simulation_results_x0=0.02_linked_low eps/",
    "simulation_results_x0=0.02_linked_eps/"
]
OUTPUT_CSV_FILE = "complexity_data_linked_eps*0.5.csv"       # 数値データの保存名
OUTPUT_3D_IMAGE = "complexity_3d_optimized_linked_eps*0.5.png"   # 画像の保存名

# 分析設定
nperseg_local = 245760 
noverlap_local = 184320
window_type = 'blackmanharris'

# ★解析開始時刻（秒）：これより前の時間は無視します
START_TIME = 0.025

# --- 【追加】フィルタリング設定 ---
# この周波数以下のデータ（DC成分やドリフト）を無視します
MIN_FREQ_THRESHOLD = 400.0  
# 指定周波数以上のパワーの合計がこれ以下なら「発振なし(無音)」とみなします
MIN_POWER_THRESHOLD = 1e-5 

def calculate_spectral_entropy_full(csv_filepath):
    """ 指定時間以降のデータかつ指定周波数以上を使ってエントロピーを計算 """
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty or 'pi' not in df or 'time' not in df:
            return 0.0
        
        # --- 【時間フィルタリング】指定時間（0.03秒）以降のデータを抽出 ---
        df_segment = df[df['time'] >= START_TIME]
        
        # データが足りない場合はスキップ
        if len(df_segment) < nperseg_local:
            # print(f"Skipping {csv_filepath}: Not enough data after {START_TIME}s")
            return 0.0
            
        sampling_rate = 1.0 / (df['time'].values[1] - df['time'].values[0])
        pi = df_segment['pi'].values 

        # スペクトログラム計算
        f, t, Sxx = spectrogram(pi, fs=sampling_rate, window=window_type, nperseg=nperseg_local, noverlap=noverlap_local)
        
        # --- 【周波数フィルタリング】指定周波数（300Hz）以上を抽出 ---
        valid_freq_indices = f >= MIN_FREQ_THRESHOLD
        
        # フィルタリング実行
        Sxx_filtered = Sxx[valid_freq_indices, :]
        
        # 有効な周波数帯域がない場合は0
        if Sxx_filtered.size == 0:
            return 0.0

        # 時間平均をとる
        mean_spectrum = np.mean(Sxx_filtered, axis=1)
        
        # --- 【無音判定】パワーが小さすぎる場合は0 ---
        if np.sum(mean_spectrum) < MIN_POWER_THRESHOLD:
            return 0.0
            
        # 正規化してエントロピー計算
        psd_norm = mean_spectrum / np.sum(mean_spectrum)
        
        ent = entropy(psd_norm, base=2)
        max_entropy = np.log2(len(psd_norm))
        
        if max_entropy == 0:
            return 0.0

        return ent / max_entropy

    except Exception as e:
        print(f"Error processing {csv_filepath}: {e}")
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

        # 丸め処理
        raw_eps = float(match.group(1))
        raw_ps = float(match.group(2))
        eps = float(f"{raw_eps:.1e}")
        ps = float(f"{raw_ps:.1e}")
        
        # ここで計算されるのは「正規化前（理論上の絶対値）」の複雑度
        raw_complexity = calculate_spectral_entropy_full(csv_filepath)
        
        results.append({
            'epsilon': eps, 
            'ps': ps, 
            'raw_complexity': raw_complexity  # 正規化前の値を保存
        })
        
        if (i+1) % 50 == 0: print(f"    Progress: {i+1} files")

df_results = pd.DataFrame(results)
df_results = df_results.drop_duplicates(subset=['epsilon', 'ps'], keep='last')

# --- 【追加】正規化後の値を計算して列に追加 ---
#max_val = df_results['raw_complexity'].max()
#if max_val > 0:
#    print(f"全体での最大エントロピー: {max_val:.4f}")
    # 最大値を1.0とする相対値を計算
#    df_results['normalized_complexity'] = df_results['raw_complexity'] / max_val
#else:
#    df_results['normalized_complexity'] = 0.0

# --- 【追加】CSVファイルへの書き出し ---
# 見やすいようにパラメータでソート
df_results = df_results.sort_values(by=['epsilon', 'ps'])

# CSV保存
print(f"数値データを {OUTPUT_CSV_FILE} に保存しています...")
df_results.to_csv(OUTPUT_CSV_FILE, index=False)
print("保存完了。")


# ---------------------------------------------------------
# 以下、グラフ描画（normalized_complexity を使用）
# ---------------------------------------------------------
print("3Dグラフのデータ準備中...")

epsilon_axis = sorted(df_results['epsilon'].unique())
ps_axis = sorted(df_results['ps'].unique())

Z_matrix = np.zeros((len(ps_axis), len(epsilon_axis)))
res_map = {}
for _, row in df_results.iterrows():
    res_map[(row['epsilon'], row['ps'])] = row['raw_complexity']
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
    current_dy = ps_diffs[i] * 0.8#0.35
    for j, eps in enumerate(epsilon_axis):
        val = Z_matrix[i, j]
        current_dx = eps_diffs[j] * 0.8#0.35
        
        x_pos.append(eps)
        y_pos.append(ps)
        z_pos.append(0)
        dx.append(current_dx)
        dy.append(current_dy)
        dz.append(val)

colors_list = ['#ffffff', '#87CEFA', '#DC143C'] 
cmap_custom = mcolors.LinearSegmentedColormap.from_list('white_to_red', colors_list)
colors = np.array([cmap_custom(h) for h in dz])

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

# ソート（描画順序の修正）
score = x_pos + y_pos 
sort_order = np.argsort(score)[::-1] 
x_pos = x_pos[sort_order]
y_pos = y_pos[sort_order]
z_pos = z_pos[sort_order]
dx = dx[sort_order]
dy = dy[sort_order]
dz = dz[sort_order]
colors = colors[sort_order]

print("描画処理中...")
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)

LABEL_FONTSIZE = 14
TICK_FONTSIZE = 10
ax.set_xlabel('Epsilon (ε)', fontsize=LABEL_FONTSIZE, labelpad=20)
ax.set_ylabel('Pressure (ps)', fontsize=LABEL_FONTSIZE, labelpad=20)
ax.set_zlabel('Complexity', fontsize=LABEL_FONTSIZE, labelpad=10)

x_ticks = epsilon_axis[::3]
ax.set_xticks(x_ticks)
ax.set_xticklabels([f"{val:.1e}" for val in x_ticks], rotation=45, fontsize=TICK_FONTSIZE)

y_ticks = ps_axis[::3]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{val:.1e}" for val in y_ticks], fontsize=TICK_FONTSIZE, rotation=-15)

ax.set_zlim(0, 1.0)
ax.view_init(elev=30, azim=-60)
plt.title('3D Complexity Map', fontsize=20)
plt.tight_layout()

plt.savefig(OUTPUT_3D_IMAGE, dpi=150)
print(f"完了: 画像を {OUTPUT_3D_IMAGE} に保存しました。")