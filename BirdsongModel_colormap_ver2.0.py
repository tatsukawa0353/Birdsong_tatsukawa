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
CAT_SINGLE_TONE = 4 # (分析用の一時カテゴリ)
# --------------------

# --- ここで設定を変更できます ---
# C++がCSVファイルを出力したフォルダ
INPUT_FOLDER = "simulation_results_2_x0=0.01/"
# 最終的に出力するパラメータマップの画像ファイル名
OUTPUT_IMAGE = "parameter_map_2_x0=0.01.png"

# 【重要】kashika.py (視覚的判断) と同じ分析設定
nperseg_local = 245760 
noverlap_local = 184320
window_type = 'blackmanharris'

# 【重要】kashika.py (視覚的判断) と同じ「音あり」のしきい値
VISUAL_THRESHOLD_DB = -38.0
# --------------------------------

# (VISUAL_THRESHOLD_DB = -38.0 などの設定の後に追加)

def classify_chunk(pi_chunk, sampling_rate):
    """ 
    渡された音声チャンクを分析し、
    「定常状態」の5カテゴリ(0,1,2,3,4)のいずれかに分類するヘルパー関数
    """
    
    # --- 0. NaNチェック ---
    if not np.isfinite(pi_chunk).all():
        return CAT_NO_SOUND
        
    # --- 1. 振幅が小さすぎる ---
    if np.std(pi_chunk) < 2.0e-3: 
        return CAT_NO_SOUND

    # --- 2. スペクトル分析の準備 ---
    if len(pi_chunk) < nperseg_local:
        return CAT_NO_SOUND 

    f, t, Sxx = spectrogram(pi_chunk, fs=sampling_rate, window=window_type, nperseg=nperseg_local, noverlap=noverlap_local)
    
    if np.max(Sxx) <= 0:
        return CAT_NO_SOUND

    # --- 3. dB変換 (正規化) ---
    Sxx_normalized = Sxx / np.max(Sxx)
    db_Sxx = 10 * np.log10(Sxx_normalized + 1e-10)

    # --- 4. 視覚的しきい値 (全体) ---
    if np.max(db_Sxx) < VISUAL_THRESHOLD_DB: # しきい値
        return CAT_NO_SOUND
    
    # --- 5. "ノイジー" の判定 (先に) ---
    Sxx_thresholded_power = np.where(db_Sxx >= VISUAL_THRESHOLD_DB, Sxx, 0)
    mean_power_thresholded_avg = np.mean(Sxx_thresholded_power, axis=1)
    
    sorted_powers = np.sort(mean_power_thresholded_avg)[::-1]
    top3_power = np.sum(sorted_powers[:3])
    total_visible_power = np.sum(sorted_powers)
    
    if total_visible_power > 0: # ゼロ除算を回避
        concentration_ratio = top3_power / total_visible_power
        if concentration_ratio < 0.38: 
            return CAT_NOISY
    # ------------------------------------

    # --- "ノイジー" ではないものだけ、ピークを探す ---
    max_spectrum_db = np.max(db_Sxx, axis=1) 
    max_spectrum_thresholded = np.where(max_spectrum_db >= VISUAL_THRESHOLD_DB, max_spectrum_db, -200)

    peaks, properties = find_peaks(max_spectrum_thresholded, height=VISUAL_THRESHOLD_DB, prominence=5) 
    
    if len(peaks) == 0: 
        return CAT_NO_SOUND
    
    # --- 【重要】ここが修正点 ---
    if len(peaks) == 1:
        return CAT_SINGLE_TONE # 単一音は (4) として返す
    
    # --- len(peaks) > 1 の場合 (倍音 or サブ) ---
    peak_freqs = f[peaks]

    # "サブハーモニクス" の判定
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

    return CAT_HARMONIC

def analyze_simulation(csv_filepath):
    """ 
    【修正】1つのCSVファイルを「前半」と「後半」に分けて分析し、
    「ノイジー→倍音」を「ノイジー」として判定、
    「単一音」を「音なし」として判定する関数
    """
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty or 'pi' not in df or 'time' not in df or len(df) < 2:
            return CAT_NO_SOUND

        sampling_rate = 1.0 / (df['time'].values[1] - df['time'].values[0])

        total_len = len(df['pi'])
        start_early = total_len // 100
        end_early = total_len // 4
        start_late = total_len * 2 // 3
        
        pi_early_chunk = df['pi'].values[start_early:end_early]
        pi_late_chunk = df['pi'].values[start_late:]

        if len(pi_early_chunk) < nperseg_local or len(pi_late_chunk) < nperseg_local:
            start_index = len(df) // 24 * 19 
            pi_full_chunk = df['pi'].values[start_index:]
            if len(pi_full_chunk) < nperseg_local:
                return CAT_NO_SOUND
            
            # --- 短いファイルも最終マッピングが必要 ---
            category_full = classify_chunk(pi_full_chunk, sampling_rate)
            if category_full == CAT_SINGLE_TONE:
                return CAT_NO_SOUND
            return category_full

        # --- 前半と後半を別々に分類 (0, 1, 2, 3, 4 のいずれかが返る) ---
        category_early = classify_chunk(pi_early_chunk, sampling_rate)
        category_late = classify_chunk(pi_late_chunk, sampling_rate)
        # ----------------------------

        # --- 【重要】時間変化の判定 ---
        
        # 1. 「ノイジー -> (倍音 or サブ or 単一音)」 の場合
        if category_early == CAT_NOISY and (category_late == CAT_HARMONIC or category_late == CAT_SUBHARMONIC or category_late == CAT_SINGLE_TONE):
            return CAT_NOISY # 「ノイジー」(3) として判定
            
        # 2. その他の変化は、より安定した「後半」の状態を代表として返す
        
        # --- 【重要】最終カテゴリへのマッピング ---
        
        # もし後半が「単一音」(4) なら、「音なし」(0) にマッピング
        if category_late == CAT_SINGLE_TONE:
            return CAT_NO_SOUND
        
        # それ以外 (0, 1, 2, 3) は、そのまま返す
        return category_late

    except Exception as e:
        print(f"  [エラー] {csv_filepath} の処理中に予期せぬエラー: {e}")
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

plt.title('Parameter Map of Birdsong Simulation_2_x0=0.01')
plt.tight_layout()

plt.savefig(OUTPUT_IMAGE)
print(f"\nパラメータマップを {OUTPUT_IMAGE} という名前で保存しました。")