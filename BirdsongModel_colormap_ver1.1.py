# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, find_peaks
import glob
import os
import re # ファイル名から数値を抽出するため
import matplotlib.colors as mcolors

# --- 【修正点】カテゴリを8種類に拡張 ---
CAT_NO_SOUND = 0    # 音出力なし
CAT_SINGLE_TONE = 1 # 単一音 (倍音なし)
CAT_HARMONIC = 2    # 倍音構造
CAT_SUBHARMONIC = 3 # サブハーモニクス
CAT_NOISY = 4       # ノイジー (カオス)
# --- 時間変化カテゴリ ---
CAT_HARMONIC_THEN_NO_SOUND = 5 # 倍音 -> 音無し (例: Image 3)
CAT_NOISY_THEN_NO_SOUND = 6    # ノイジー -> 音無し (例: Image 1, 2)
CAT_NOISY_THEN_HARMONIC = 7    # ノイジー -> 倍音 (例: Image 4)
# ------------------------------------

# --- ここで設定を変更できます ---
# C++がCSVファイルを出力したフォルダ
INPUT_FOLDER = "simulation_results_2_x0=0.02/"
# 最終的に出力するパラメータマップの画像ファイル名
OUTPUT_IMAGE = "parameter_map_2_x0=0.02_ver1.1.png"

# kashika.py と同じ分析設定
nperseg_local = 245760 
noverlap_local = 184320
window_type = 'blackmanharris'

# 「音あり」とみなすしきい値 (あなたの視覚的基準)
VISUAL_THRESHOLD_DB = -38.0
# --------------------------------

def classify_chunk(pi_chunk, sampling_rate):
    """ 
    【新機能】渡された音声チャンクを分析し、
    「定常状態」の5カテゴリのいずれかに分類するヘルパー関数
    """
    
    # --- 1. "音出力なし" の判定 (振幅) ---
    #if np.std(pi_chunk) < 1e-7: # しきい値 1
        #return CAT_NO_SOUND

    # --- 2. スペクトル分析の準備 ---
    if len(pi_chunk) < nperseg_local:
        return CAT_NO_SOUND # 信号が短すぎて分析できない

    f, t, Sxx = spectrogram(pi_chunk, fs=sampling_rate, window=window_type, nperseg=nperseg_local, noverlap=noverlap_local)
    
    if np.max(Sxx) <= 0:
        return CAT_NO_SOUND

    # --- 3. kashika.py と同じ正規化 + dB変換 ---
    Sxx_normalized = Sxx / np.max(Sxx)
    db_Sxx = 10 * np.log10(Sxx_normalized + 1e-10)

    # --- 4. 視覚的しきい値で「音なし」を判定 ---
    if np.max(db_Sxx) < VISUAL_THRESHOLD_DB: # しきい値 2
        return CAT_NO_SOUND
    
    # --- 5. 「音あり」と判定されたものを、さらに分類 ---
    mean_spectrum_db = np.mean(db_Sxx, axis=1) # 時間平均
    mean_spectrum_thresholded = np.where(mean_spectrum_db >= VISUAL_THRESHOLD_DB, mean_spectrum_db, -200)

    peaks, properties = find_peaks(mean_spectrum_thresholded, height=VISUAL_THRESHOLD_DB, prominence=0.01) # しきい値 3
    
    if len(peaks) == 0: 
        return CAT_NO_SOUND
    
    # --- 【修正点】単一音 (Image 5) を検出 ---
    if len(peaks) == 1:
        return CAT_SINGLE_TONE 
    # ---------------------------------------

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
    
    if concentration_ratio < 0.37: # しきい値 4 2_x0=0.02のみ　通常0.372
        return CAT_NOISY
    
    # 複数ピークがあり、サブでもノイジーでもなければ「倍音構造」
    return CAT_HARMONIC


def analyze_simulation_with_time(csv_filepath):
    """ 
    【修正点】1つのCSVファイルを「前半」と「後半」に分けて分析し、
    時間変化を含む8カテゴリのいずれかに分類するメイン関数
    """
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty or 'pi' not in df or 'time' not in df or len(df) < 2:
            return CAT_NO_SOUND

        sampling_rate = 1.0 / (df['time'].values[1] - df['time'].values[0])

        if not np.isfinite(df['pi']).all():
            return CAT_NO_SOUND # NaNやinfは「音なし」
        
        # --- 【修正点】信号を「前半」と「後半」に分割 ---
        # 最初の10％をカットで10→33％を「前半」、最後の2/3を「後半」として分析
        total_len = len(df['pi'])
        start_early = total_len // 30 
        end_early = total_len  // 4
        start_late = total_len * 2 // 3
        
        pi_early_chunk = df['pi'].values[start_early:end_early]
        pi_late_chunk = df['pi'].values[start_late:]
        # ---------------------------------------------

        # 各チャンクの長さが分析可能かチェック
        if len(pi_early_chunk) < nperseg_local or len(pi_late_chunk) < nperseg_local:
             # 短すぎる場合は、古いロジック（最後の1/3）で全体を判定
             start_index = len(df) #* 2 // 3
             pi_full_chunk = df['pi'].values#[start_index:]
             if len(pi_full_chunk) < nperseg_local:
                 return CAT_NO_SOUND
             return classify_chunk(pi_full_chunk, sampling_rate)

        # --- 前半と後半を別々に分類 ---
        category_early = classify_chunk(pi_early_chunk, sampling_rate)
        category_late = classify_chunk(pi_late_chunk, sampling_rate)
        # ----------------------------

        # --- 【修正点】時間変化のパターンを検出 ---
        
        # 安定状態 (前半も後半も同じカテゴリ)
        if category_early == category_late:
            return category_late # 最終的な安定状態を返す
            
        # 1. ノイジー -> 音無し (Image 1, 2)
        if category_early == CAT_NOISY and category_late == CAT_NO_SOUND:
            return CAT_NOISY_THEN_NO_SOUND
            
        # 2. 倍音 -> 音無し (Image 3)
        # (単一音か倍音構造かは区別せず「音あり」として扱う)
        if (category_early == CAT_HARMONIC or category_early == CAT_SINGLE_TONE) and category_late == CAT_NO_SOUND:
            return CAT_HARMONIC_THEN_NO_SOUND

        # 3. ノイジー -> 倍音 (Image 4)
        if category_early == CAT_NOISY and (category_late == CAT_HARMONIC or category_late == CAT_SINGLE_TONE):
            return CAT_NOISY_THEN_HARMONIC
            
        # 4. その他の変化は、より安定した「後半」の状態を代表として返す
        return category_late

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
category_names = {
    0: '音なし', 1: '単一音', 2: '倍音', 3: 'サブ', 4: 'ノイズ',
    5: '倍音->音なし', 6: 'ノイズ->音なし', 7: 'ノイズ->倍音'
}

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
    
    category = analyze_simulation_with_time(csv_filepath)
    results[(eps, ps)] = category
    print(f"  ({i+1}/{len(csv_files)}) eps={eps:.1e}, ps={ps:.1e} -> {category_names.get(category, '不明')}")

print("すべてのファイルの分析が完了しました。")

# 3. マトリックスの作成
epsilon_axis = sorted(list(epsilon_set))
ps_axis = sorted(list(ps_set))

result_matrix = np.zeros((len(ps_axis), len(epsilon_axis)))

for i, ps in enumerate(ps_axis):
    for j, eps in enumerate(epsilon_axis):
        result_matrix[i, j] = results.get((eps, ps), CAT_NO_SOUND)

# 4. 【修正点】カラーマップと凡例を8カテゴリに対応
cmap = mcolors.ListedColormap([
    '#ffffff', # 0: 音なし (白)
    '#a9a9a9', # 1: 単一音 (濃いグレー)
    '#87CEFA', # 2: 倍音 (水色)
    '#FF7F50', # 3: サブ (オレンジ)
    '#DC143C', # 4: ノイズ (赤)
    '#b0e0e6', # 5: 倍音->音なし (薄い水色)
    '#ffcccb', # 6: ノイズ->音なし (薄い赤)
    '#ffdab9'  # 7: ノイズ->倍音 (薄いオレンジ)
])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(12, 10))
plt.imshow(result_matrix, aspect='auto', origin='lower', cmap=cmap, norm=norm,
           extent=[epsilon_axis[0], epsilon_axis[-1], ps_axis[0], ps_axis[-1]])

plt.xlabel('Epsilon (ε)')
plt.ylabel('Pressure (ps)')

plt.xticks(epsilon_axis, [f"{eps:.1e}" for eps in epsilon_axis], rotation=45)
plt.yticks(ps_axis, [f"{ps:.1e}" for ps in ps_axis])

cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6, 7])
cbar.set_ticklabels([
    'No Sound', 'Single Tone', 'Harmonic', 'Subharmonic', 'Noisy',
    'Harmonic -> No Sound', 'Noisy -> No Sound', 'Noisy -> Harmonic'
])
cbar.set_label('Vibration Type')
# ----------------------------------------------------

plt.title('Parameter Map of Birdsong Simulation_2_x0=0.02')
plt.tight_layout()

plt.savefig(OUTPUT_IMAGE)
print(f"\nパラメータマップを {OUTPUT_IMAGE} という名前で保存しました。")