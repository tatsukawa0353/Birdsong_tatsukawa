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
OUTPUT_CSV_FILE = "complexity_data_db_fixed_1.csv"
OUTPUT_3D_IMAGE = "complexity_3d_db_fixed_1.png"

# 分析設定
nperseg_local = 245760 
noverlap_local = 184320
window_type = 'blackmanharris'

# ★ダイナミックレンジの設定（dB）
# 0dB（最大）からどれくらい下までを評価対象にするか
# 一般的なオーディオ分析では 60〜80dB 程度が妥当です
DYNAMIC_RANGE_DB = 38.002 

def calculate_db_entropy(csv_filepath):
    """
    最大値を0dBに正規化したスペクトルを用いてエントロピーを計算
    """
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty or 'pi' not in df or len(df) < nperseg_local:
            return 0.0
        
        sampling_rate = 1.0 / (df['time'].values[1] - df['time'].values[0])
        pi = df['pi'].values 
        if len(pi) < nperseg_local: return 0.0

        # 1. スペクトル計算 (Linear)
        f, t, Sxx = spectrogram(pi, fs=sampling_rate, window=window_type, nperseg=nperseg_local, noverlap=noverlap_local)
        mean_spectrum_linear = np.mean(Sxx, axis=1)
        
        peak_power = np.max(mean_spectrum_linear)
        if peak_power == 0: return 0.0

        # 2. 最大値を 0dB に正規化 (kashika.py と同じ)
        # S_norm = S / max(S)  -> log10(S_norm) は max 0
        norm_spectrum_linear = mean_spectrum_linear / peak_power
        
        # 0割りを防ぐため微小値を加算してからdB変換
        epsilon = 1e-20
        spectrum_db = 10 * np.log10(norm_spectrum_linear + epsilon)
        
        # 3. ダイナミックレンジによる足切りとシフト
        # -80dB より小さい値は -80dB にクリップし、全体を +80 して正の値にする
        # 結果: 0 (無音/ノイズ) 〜 80 (ピーク) の範囲になる
        spectrum_shifted = np.maximum(spectrum_db, -DYNAMIC_RANGE_DB) + DYNAMIC_RANGE_DB
        
       # 高周波成分（インデックスが大きいほう）の値を小さくして、倍音の影響を下げる
        weights = 1.0 / (np.log1p(np.arange(len(spectrum_shifted))) + 1.0)
        spectrum_weighted = spectrum_shifted * weights
        # ---------------------------

        # 4. 確率分布化
        # spectrum_shifted ではなく spectrum_weighted を使う
        if np.sum(spectrum_weighted) == 0: return 0.0
        psd_norm = spectrum_weighted / np.sum(spectrum_weighted)
        
        # 5. エントロピー計算
        ent = entropy(psd_norm, base=2)
        max_entropy = np.log2(len(psd_norm))
        
        return ent / max_entropy

    except Exception:
        return 0.0

# --- メイン処理 ---
print(f"解析開始 (0dB Normalized, Range={DYNAMIC_RANGE_DB}dB)...")

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
        eps = float(f"{raw_eps:.1e}")
        ps = float(f"{raw_ps:.1e}")
        
        # 新しい関数を使用
        raw_complexity = calculate_db_entropy(csv_filepath)
        
        results.append({'epsilon': eps, 'ps': ps, 'raw_complexity': raw_complexity})
        if (i+1) % 50 == 0: print(f"    Progress: {i+1} files")

df_results = pd.DataFrame(results)
df_results = df_results.drop_duplicates(subset=['epsilon', 'ps'], keep='last')

# 正規化 (全体Max=1.0)
max_val = df_results['raw_complexity'].max()
if max_val > 0:
    df_results['normalized_complexity'] = df_results['raw_complexity'] / max_val
else:
    df_results['normalized_complexity'] = 0.0

# ソートとCSV保存
df_results = df_results.sort_values(by=['epsilon', 'ps'])

df_export = df_results.copy()
df_export['epsilon'] = df_export['epsilon'].map(lambda x: f"{x:.1e}")
df_export['ps'] = df_export['ps'].map(lambda x: f"{x:.1e}")
df_export['raw_complexity'] = df_export['raw_complexity'].map(lambda x: f"{x:.4f}")
df_export['normalized_complexity'] = df_export['normalized_complexity'].map(lambda x: f"{x:.4f}")

df_export.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"CSV保存完了: {OUTPUT_CSV_FILE}")

# --- グラフ描画 ---
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

#min_step = np.min(eps_diffs[eps_diffs > 0])
#max_allowed_width = min_step * 3.0

for i, ps in enumerate(ps_axis):
    current_dy = ps_diffs[i] * 0.8
    for j, eps in enumerate(epsilon_axis):
        val = Z_matrix[i, j]
        # 横幅キャップ
        raw_width = eps_diffs[j] * 0.8
        current_dx = raw_width
        
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

# マスク
mask = dz > 0.001
x_pos = x_pos[mask]
y_pos = y_pos[mask]
z_pos = z_pos[mask]
dx = dx[mask]
dy = dy[mask]
dz = dz[mask]

# 色設定
colors_list = ['#ffffff', "#5CB4EB", "#CA0C32"] 
cmap_custom = mcolors.LinearSegmentedColormap.from_list('white_to_red', colors_list)
colors = np.array([cmap_custom(h) for h in dz])

# ソート (奥→手前)
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
ax.set_zlabel('Relative Complexity', fontsize=LABEL_FONTSIZE, labelpad=10)

x_ticks = epsilon_axis[::3]
ax.set_xticks(x_ticks)
ax.set_xticklabels([f"{val:.1e}" for val in x_ticks], rotation=45, fontsize=TICK_FONTSIZE)

y_ticks = ps_axis[::3]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{val:.1e}" for val in y_ticks], fontsize=TICK_FONTSIZE, rotation=-15)

ax.set_zlim(0, 1.0)
ax.view_init(elev=30, azim=-60)
plt.title('3D Complexity Map (0dB Normalized)', fontsize=20)
plt.tight_layout()

print("保存中...")
plt.savefig(OUTPUT_3D_IMAGE, dpi=150)
print(f"完了: {OUTPUT_3D_IMAGE}")