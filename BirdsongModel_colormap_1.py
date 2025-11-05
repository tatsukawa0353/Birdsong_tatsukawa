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
INPUT_FOLDER = "simulation_results_1_x0=0.02/"
# 最終的に出力するパラメータマップの画像ファイル名
OUTPUT_IMAGE = "parameter_map_1_x0=0.02.png"

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

        if not np.isfinite(df['pi']).all():
            return CAT_NO_SOUND # NaNやinfは「音なし」
        
        start_index = len(df) // 3 * 2 #解析範囲（この場合後半1/3のみ）
        pi = df['pi'].values[start_index:]

        if np.std(pi) < 1e-4: # 振幅がほぼゼロなら、計算するまでもなく「音なし」
            return CAT_NO_SOUND

        if len(pi) < nperseg_local:
            return CAT_NO_SOUND # 定常状態が短すぎて分析できない

        # --- 1. kashika.py と同じスペクトログラムを計算 ---
        # 【修正点】spectrogramは f, t, Sxx の3つの値を返すため、t も受け取る
        f, t, Sxx = spectrogram(pi, fs=sampling_rate, window=window_type, nperseg=nperseg_local, noverlap=noverlap_local)
        
        if np.max(Sxx) <= 0:
            return CAT_NO_SOUND

        # --- 2. kashika.py と同じ正規化 + dB変換 ---
        Sxx_normalized = Sxx / np.max(Sxx)
        db_Sxx = 10 * np.log10(Sxx_normalized + 1e-10)

        # --- 3. 【最重要】視覚的しきい値で「音なし」を判定 ---
        if np.max(db_Sxx) < VISUAL_THRESHOLD_DB:
            return CAT_NO_SOUND
        
        # --- 4. 「音あり」と判定されたものを、さらに分類 ---
        
        mean_spectrum_db = np.mean(db_Sxx, axis=1) # 時間平均
        mean_spectrum_thresholded = np.where(mean_spectrum_db >= VISUAL_THRESHOLD_DB, mean_spectrum_db, -200)

        peaks, properties = find_peaks(mean_spectrum_thresholded, height=VISUAL_THRESHOLD_DB, prominence=35) #調整ポイント5→
        
        if len(peaks) == 0: 
            return CAT_NO_SOUND
        if len(peaks) == 1:
            return CAT_HARMONIC 

        peak_freqs = f[peaks]

        # "サブハーモニクス" の判定を先に
        valid_peak_freqs = peak_freqs[peak_freqs > 100]
        if len(valid_peak_freqs) == 0:
             pass 
        else:
            f0 = np.min(valid_peak_freqs)
            subharmonic_freq_min = f0 * 0.40
            subharmonic_freq_max = f0 * 0.60
            
            has_subharmonic = False
            for freq in peak_freqs:
                if subharmonic_freq_min < freq < subharmonic_freq_max:
                    has_subharmonic = True
                    break
            
            if has_subharmonic:
                return CAT_SUBHARMONIC 

        # "ノイジー" の判定
        Sxx_thresholded_power = np.where(db_Sxx >= VISUAL_THRESHOLD_DB, Sxx, 0)
        mean_power_thresholded_avg = np.mean(Sxx_thresholded_power, axis=1)
        
        sorted_powers = np.sort(mean_power_thresholded_avg)[::-1]
        top3_power = np.sum(sorted_powers[:3])
        total_visible_power = np.sum(sorted_powers)
        
        if total_visible_power <= 0:
             return CAT_HARMONIC

        concentration_ratio = top3_power / total_visible_power
        
        if concentration_ratio < 0.38: #調整ポイント
            return CAT_NOISY
        
        return CAT_HARMONIC

    except Exception as e:
        print(f"  [エラー] {csv_filepath} の処理中に予期せぬエラー: {e}")
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

plt.title('Parameter Map of Birdsong Simulation_1_x0=0.02')
plt.tight_layout()

plt.savefig(OUTPUT_IMAGE)
print(f"\nパラメータマップを {OUTPUT_IMAGE} という名前で保存しました。")