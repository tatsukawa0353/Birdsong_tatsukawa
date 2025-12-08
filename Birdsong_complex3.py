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
OUTPUT_CSV_FILE = "complexity_data_weighted_1.csv" # ソート済みCSV
OUTPUT_3D_IMAGE = "complexity_3d_weighted_1.png"

# 分析設定
nperseg_local = 245760 
noverlap_local = 184320
window_type = 'blackmanharris'

# ★解析開始時刻（秒）
START_TIME = 0.03 #0.01 one bronchus

# --- 【追加】フィルタリング設定 ---
# この周波数以下のデータ（DC成分やドリフト）を無視します
MIN_FREQ_THRESHOLD = 100  #150 one bronchus
# 指定周波数以上のパワーの合計がこれ以下なら「発振なし(無音)」とみなします
MIN_POWER_THRESHOLD = 1e-5

def calculate_weighted_spectral_entropy(csv_filepath):
    """ 重み付けスペクトル・エントロピー (低周波カット機能付き) """
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty or 'pi' not in df or len(df) < nperseg_local:
            return 0.0
        
        sampling_rate = 1.0 / (df['time'].values[1] - df['time'].values[0])
        
        # --- 指定時間以降のデータを抽出 ---
        df_segment = df[df['time'] >= START_TIME]
        pi = df_segment['pi'].values

        if len(pi) < nperseg_local:
            return 0.0

        # スペクトログラム計算
        f, t, Sxx = spectrogram(pi, fs=sampling_rate, window=window_type, nperseg=nperseg_local, noverlap=noverlap_local)
        
        # --- 【追加】周波数フィルタリング処理 ---
        # 指定した周波数(MIN_FREQ_THRESHOLD)以上のインデックスを取得
        valid_freq_indices = f >= MIN_FREQ_THRESHOLD
        
        # フィルタリング後の周波数成分だけを取り出す
        f_filtered = f[valid_freq_indices]
        Sxx_filtered = Sxx[valid_freq_indices, :] # スペクトログラムも対応する行だけ残す
        
        # もし有効な周波数帯域がなくなってしまったら0を返す
        if len(f_filtered) == 0:
            return 0.0

        # 時間平均をとる (フィルタ済みのデータで)
        mean_spectrum = np.mean(Sxx_filtered, axis=1)
        
        # --- 【追加】パワー閾値判定 ---
        # ノイズのみの「無音状態」が高エントロピーになるのを防ぐため、
        # フィルタ後の総パワーが小さすぎる場合は複雑性を0とする
        if np.sum(mean_spectrum) < MIN_POWER_THRESHOLD:
            return 0.0
            
        # 重み付け (高周波ほど値を小さく評価)
        # ※フィルタリング後の配列に対して重み付けを行います
        weights = 1.0 / (np.log1p(np.arange(len(mean_spectrum))) + 1.0)
        weighted_spectrum = mean_spectrum * weights
        
        # 正規化
        spectrum_sum = np.sum(weighted_spectrum)
        if spectrum_sum == 0:
            return 0.0
            
        psd_norm = weighted_spectrum / spectrum_sum
        
        ent = entropy(psd_norm, base=2)
        max_entropy = np.log2(len(psd_norm))
        
        if max_entropy == 0:
            return 0.0
            
        return ent / max_entropy

    except Exception as e:
        print(f"Error in {csv_filepath}: {e}") # デバッグ用にエラー表示
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

        # 丸め処理
        raw_eps = float(match.group(1))
        raw_ps = float(match.group(2))
        eps = float(f"{raw_eps:.1e}")
        ps = float(f"{raw_ps:.1e}")
        
        raw_complexity = calculate_weighted_spectral_entropy(csv_filepath)
        
        results.append({'epsilon': eps, 'ps': ps, 'raw_complexity': raw_complexity})
        if (i+1) % 50 == 0: print(f"    Progress: {i+1} files")

df_results = pd.DataFrame(results)
df_results = df_results.drop_duplicates(subset=['epsilon', 'ps'], keep='last')

# 正規化
max_val = df_results['raw_complexity'].max()
if max_val > 0:
    df_results['normalized_complexity'] = df_results['raw_complexity'] / max_val
else:
    df_results['normalized_complexity'] = 0.0

# --- 【重要】ソート処理 ---
# Epsilonの昇順 -> Pressureの昇順 に並び替え
df_results = df_results.sort_values(by=['epsilon', 'ps'])

# --- 【重要】CSV出力用の整形 ---
# 計算用の数値データとは別に、表示用のデータフレームを作る
df_export = df_results.copy()
# 科学的記数法 (1.00e+07) に変換して見やすくする
df_export['epsilon'] = df_export['epsilon'].map(lambda x: f"{x:.1e}")
df_export['ps'] = df_export['ps'].map(lambda x: f"{x:.1e}")
df_export['raw_complexity'] = df_export['raw_complexity'].map(lambda x: f"{x:.4f}")
df_export['normalized_complexity'] = df_export['normalized_complexity'].map(lambda x: f"{x:.4f}")

print(f"整形済みデータを {OUTPUT_CSV_FILE} に保存しています...")
df_export.to_csv(OUTPUT_CSV_FILE, index=False)
print("保存完了。")


# --- グラフ描画 (数値データを使用) ---
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
        current_dx = eps_diffs[j] * 0.8
        
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

# 描画順ソート (奥→手前)
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
plt.title('3D Complexity Map (Spectral Entropy)', fontsize=20)
plt.tight_layout()

print("保存中...")
plt.savefig(OUTPUT_3D_IMAGE, dpi=150)
print(f"完了: {OUTPUT_3D_IMAGE}")