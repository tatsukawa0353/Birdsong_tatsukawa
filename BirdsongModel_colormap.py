# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, find_peaks
import glob
import os
import re # ファイル名から数値を抽出するため
import matplotlib.colors as mcolors

# --- カテゴリ定義 ---
CAT_NO_SOUND = 0    # 音出力なし
CAT_HARMONIC = 1    # 倍音構造 (通常の音)
CAT_SUBHARMONIC = 2 # サブハーモニクス
CAT_NOISY = 3       # ノイジー (カオス)
# --------------------

# --- ここで設定を変更できます ---
# C++がCSVファイルを出力したフォルダ
INPUT_FOLDER = "simulation_results_1_x0=0.04/"
# 最終的に出力するパラメータマップの画像ファイル名
OUTPUT_IMAGE = "parameter_map_1_x0=0.04.png"

# 【重要】kashika.py (視覚的判断) と同じ分析設定
nperseg_local = 245760 
noverlap_local = 184320
window_type = 'blackmanharris'

# 【重要】kashika.py (視覚的判断) と同じ「音あり」のしきい値
VISUAL_THRESHOLD_DB = -38.0
# --------------------------------

def analyze_simulation(csv_filepath):
    """ 
    1つのCSVファイルを分析し、4つのカテゴリのいずれかに分類する関数 
    """
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty or 'pi' not in df or 'time' not in df or len(df) < 2:
            return CAT_NO_SOUND

        sampling_rate = 1.0 / (df['time'].values[1] - df['time'].values[0])

        if df['pi'].isnull().any():
            return CAT_NO_SOUND # NaNは「音なし」

        start_index = len(df) // 3 * 2
        pi = df['pi'].values[start_index:]

        if np.std(pi) < 1e-4: # 振幅がほぼゼロなら、計算するまでもなく「音なし」
            return CAT_NO_SOUND

        if len(pi) < nperseg_local:
            return CAT_NO_SOUND # 定常状態が短すぎて分析できない

        # --- 1. kashika.py と同じスペクトログラムを計算 ---
        f, Sxx = spectrogram(pi, fs=sampling_rate, window=window_type, nperseg=nperseg_local, noverlap=noverlap_local)
        
        # Sxxがゼロ（無音）の場合
        if np.max(Sxx) <= 0:
            return CAT_NO_SOUND

        # --- 2. kashika.py と同じ正規化 + dB変換 ---
        Sxx_normalized = Sxx / np.max(Sxx)
        db_Sxx = 10 * np.log10(Sxx_normalized + 1e-10)

        # --- 3. 【最重要】kashika.py の視覚的しきい値で「音なし」を判定 ---
        if np.max(db_Sxx) < VISUAL_THRESHOLD_DB:
            # 最大音量が-38dBより弱い場合、視覚的には「真っ白」と判断される
            return CAT_NO_SOUND
        
        # --- 4. 「音あり」と判定されたものを、さらに分類 ---
        
        mean_spectrum_db_normalized = np.mean(db_Sxx, axis=1) # 時間平均
        
        peaks, properties = find_peaks(mean_spectrum_db_normalized, height=-50, prominence=5)
        
        if len(peaks) < 2:
            return CAT_HARMONIC # 音はあるが、ピークが1つだけなら倍音

        peak_freqs = f[peaks]
        peak_heights_db = properties['peak_heights']

        # "ノイジー" の判定
        mean_spectrum_power = np.mean(Sxx, axis=1) # パワー（dBではない）で計算
        sorted_powers = np.sort(mean_spectrum_power)[::-1]
        top3_power = np.sum(sorted_powers[:3])
        total_power = np.sum(sorted_powers)
        concentration_ratio = top3_power / total_power
        
        if concentration_ratio < 0.6: 
            return CAT_NOISY

        # "サブハーモニクス" vs "倍音構造" の判定
        valid_peak_freqs = peak_freqs[peak_freqs > 100]
        if len(valid_peak_freqs) == 0:
            return CAT_HARMONIC
            
        f0 = np.min(valid_peak_freqs)
        
        subharmonic_freq_min = f0 * 0.40
        subharmonic_freq_max = f0 * 0.60
        
        has_subharmonic = False
        for freq, height in zip(peak_freqs, peak_heights_db):
            if subharmonic_freq_min < freq < subharmonic_freq_max:
                if height > -60: # 検出感度は-50dBに設定
                    has_subharmonic = True
                    break
        
        if has_subharmonic:
            return CAT_SUBHARMONIC
        else:
            return CAT_HARMONIC

    except Exception as e:
        print(f"  [エラー] {csv_filepath} の分析中にエラー: {e}")
        return CAT_NO_SOUND # エラー時は音なし扱い

# --- メイン処理 ---
print(f"'{INPUT_FOLDER}' 内のCSVファイルの分析を開始します...")

# 1. ファイルリストを取得
csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
if not csv_files:
    print(f"エラー: '{INPUT_FOLDER}' にCSVファイルが見つかりません。")
    print("C++のシミュレーションを先に実行してください。")
    exit()

print(f"{len(csv_files)} 個のファイルが見つかりました。")

# 2. ファイル名からパラメータを抽出し、分析結果を保存する辞書
results = {}
epsilon_set = set()
ps_set = set()

pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")

for i, csv_filepath in enumerate(csv_files):
    filename = os.path.basename(csv_filepath)
    match = pattern.match(filename)
    
    if not match:
        print(f"  [警告] スキップ: {filename} のファイル名がパターンに一致しません。")
        continue

    eps = float(match.group(1))
    ps = float(match.group(2))
    
    epsilon_set.add(eps)
    ps_set.add(ps)
    
    category = analyze_simulation(csv_filepath)
    results[(eps, ps)] = category
    print(f"  ({i+1}/{len(csv_files)}) eps={eps:.1e}, ps={ps:.1e} -> {['音なし', '倍音', 'サブ', 'ノイズ'][category]}")

print("すべてのファイルの分析が完了しました。")

# 3. マトリックスの作成
epsilon_axis = sorted(list(epsilon_set))
ps_axis = sorted(list(ps_set))

result_matrix = np.zeros((len(ps_axis), len(epsilon_axis)))

for i, ps in enumerate(ps_axis):
    for j, eps in enumerate(epsilon_axis):
        result_matrix[i, j] = results.get((eps, ps), CAT_NO_SOUND)

# 4. カラーマップの描画
cmap = mcolors.ListedColormap(['#ffffff', '#87CEFA', '#FF7F50', '#DC143C'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(12, 10))
plt.imshow(result_matrix, aspect='auto', origin='lower', cmap=cmap, norm=norm,
           extent=[epsilon_axis[0], epsilon_axis[-1], ps_axis[0], ps_axis[-1]])

plt.xlabel('Epsilon (ε)')
plt.ylabel('Pressure (ps)')

plt.xticks(epsilon_axis, [f"{eps:.1e}" for eps in epsilon_axis], rotation=45)
plt.yticks(ps_axis, [f"{ps:.1e}" for ps in ps_axis])

cbar = plt.colorbar(ticks=[0, 1, 2, 3])
cbar.set_ticklabels(['No Sound', 'Harmonic', 'Subharmonic', 'Noisy'])
cbar.set_label('Vibration Type')

plt.title('Parameter Map of Birdsong Simulation_1_x0=0.04')
plt.tight_layout()

plt.savefig(OUTPUT_IMAGE)
print(f"\nパラメータマップを {OUTPUT_IMAGE} という名前で保存しました。")