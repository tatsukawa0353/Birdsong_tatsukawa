import pandas as pd
import numpy as np
from scipy.signal import find_peaks, welch, butter, filtfilt, correlate
import glob
import os
import re

# ==========================================
# ★★★ 設定エリア ★★★
# ==========================================

# 解析セットの定義（前回のものを引き継ぎ）
ANALYSIS_SETS = [
    # --- セット1 ---
    #{
     #   "input_folders": [
      #      "simulation_results_1_f0=0.1e7_x0=0.02_low epsilon/",
       #     "simulation_results_1_f0=0.1e7_x0=0.02/"
        #],
        #"output_csv": "sci_data_1_f0=0.1e7.csv",
    #},
    {
        "input_folders": [
            "simulation_results_1_x0=0.02_low parameters epsilon/",
            "simulation_results_1_x0=0.02/"
        ],
        "output_csv": "sci_data_1_f0=1.0e7.csv",
    },
    #{
        #"input_folders": [
         #  "simulation_results_2_x0=0.02_low parameters epsilon/",
         #  "simulation_results_2_x0=0.02/"
        #],
       # "output_csv": "sci_data_2.csv",
    #},
    # 他のセットも同様に追加してください...
    # (ここでは例として1つだけにしていますが、元のリストをそのままコピペしてOKです)
]

# 分析パラメータ
START_TIME = 0.05          # 解析開始時刻（秒）
CALC_MIN_FREQ = 250.0     # 計算下限周波数（Hz）
CALC_MAX_FREQ = 12000.0   # 計算上限周波数（Hz）
JITTER_THRESHOLD = 3.0    # 許容する最大ジッタ（%）。これより揺らぎが大きい音はノイズとして除外
F0_MAX_FREQ = 3500.0      # F0を探索する上限周波数

# ==========================================
# 関数定義（ジッタ計算 + SCI計算）
# ==========================================

def calculate_sci_and_jitter_robust(csv_filepath):
    """ 
    自己相関法とローパスフィルタを用いた堅牢なF0・Jitter・SCI計算
    """
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty or 'pi' not in df or 'time' not in df:
            return None
        
        df_segment = df[df['time'] >= START_TIME]
        if len(df_segment) < 1000:
            return None
            
        time_arr = df_segment['time'].values
        pi = df_segment['pi'].values 
        fs = 1.0 / (time_arr[1] - time_arr[0])

        # =========================================================
        # ★ ここに安全装置（NaN / inf チェック）を追加！
        # =========================================================
        if np.isnan(pi).any() or np.isinf(pi).any():
            return {
                'jitter': np.nan, 
                'f_AFF': np.nan, 
                'f_MSF': np.nan, 
                'SCI': np.nan, 
                'status': 'Simulation_Diverged' # 発散して壊れたデータという印をつける
            }

        # 直流成分（オフセット）を除去
        pi_norm = pi - np.mean(pi)

        # ---------------------------------------------------------
        # ★ステップ1: 自己相関（Autocorrelation）による真のF0探索
        # ---------------------------------------------------------
        corr = correlate(pi_norm, pi_norm, mode='full')
        corr = corr[len(corr)//2:] 

        # 探索上限を F0_MAX_FREQ に制限する！
        min_lag = int(fs / F0_MAX_FREQ)
        max_lag = int(fs / CALC_MIN_FREQ)
        
        if len(corr) < max_lag:
            return None

        valid_corr = corr[min_lag:max_lag]
        best_lag_idx = np.argmax(valid_corr)
        true_lag = min_lag + best_lag_idx
        true_f0 = fs / true_lag

        # ---------------------------------------------------------
        # ★ステップ2: ローパスフィルタによる波形の平滑化
        # ---------------------------------------------------------
        # F0の1.5倍以上の周波数（倍音成分）をカットするフィルタを作成
        nyq = 0.5 * fs
        cutoff = min(true_f0 * 1.5, nyq * 0.99) # ナイキスト周波数を超えないよう保護
        b, a = butter(4, cutoff / nyq, btype='low')
        
        # filtfiltを使うことで、位相（ピークのタイミング）をずらさずにフィルタリング
        pi_clean = filtfilt(b, a, pi_norm)

        # ---------------------------------------------------------
        # ★ステップ3: ジッタ (Jitter) の計算
        # ---------------------------------------------------------
        peaks, _ = find_peaks(pi_clean, distance=true_lag * 0.7)
        
        if len(peaks) < 5:
            return {'jitter': np.nan, 'f_AFF': np.nan, 'f_MSF': np.nan, 'SCI': np.nan, 'status': 'No_Peaks'}

        periods = np.diff(peaks) / fs
        mean_period = np.mean(periods)
        
        # 移動平均を用いたRAPジッタの計算（スイープを相殺）
        smoothed_periods = np.convolve(periods, np.ones(3)/3.0, mode='valid')
        mean_diff_rap = np.mean(np.abs(periods[1:-1] - smoothed_periods))
        jitter_percent = (mean_diff_rap / mean_period) * 100.0

        if jitter_percent > JITTER_THRESHOLD:
            return {'jitter': jitter_percent, 'f_AFF': true_f0, 'f_MSF': np.nan, 'SCI': np.nan, 'status': 'High_Jitter'}

        # ---------------------------------------------------------
        # ★ステップ4: SCI の計算
        # ---------------------------------------------------------
        f, Pxx = welch(pi_norm, fs=fs, nperseg=245760, window='blackmanharris')
        
        valid_indices = (f >= CALC_MIN_FREQ) & (f <= CALC_MAX_FREQ)
        f_valid = f[valid_indices]
        Pxx_valid = Pxx[valid_indices]
        
        if f_valid.size == 0 or np.sum(Pxx_valid) == 0:
            return {'jitter': jitter_percent, 'f_AFF': true_f0, 'f_MSF': np.nan, 'SCI': np.nan, 'status': 'No_Power'}

        f_AFF = true_f0
        f_MSF = np.sum(f_valid * Pxx_valid) / np.sum(Pxx_valid)
        sci = f_MSF / f_AFF

        return {
            'jitter': jitter_percent, 
            'f_AFF': f_AFF, 
            'f_MSF': f_MSF, 
            'SCI': sci, 
            'status': 'OK'
        }

    except Exception as e:
        print(f"Error processing {csv_filepath}: {e}")
        return None

# ==========================================
# ★★★ メイン処理 ★★★ (ここが抜けていました！)
# ==========================================

print(f"解析モード: 堅牢版 SCI解析 ＆ Jitterフィルタリング")
pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")

for set_idx, config in enumerate(ANALYSIS_SETS):
    target_folders = config["input_folders"]
    output_csv = config["output_csv"]

    print("\n" + "="*60)
    print(f"解析セット {set_idx + 1} を処理中...")
    print(f"  入力フォルダ数: {len(target_folders)}")
    print(f"  出力CSV: {output_csv}")
    print("="*60)

    results = [] 

    for folder in target_folders:
        if not os.path.exists(folder):
            print(f"  [警告] フォルダが見つかりません: {folder}")
            continue
        
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        print(f"  フォルダ読み込み: {folder} ({len(csv_files)} files)")
        
        for i, csv_filepath in enumerate(csv_files):
            filename = os.path.basename(csv_filepath)
            match = pattern.match(filename)
            if not match: continue

            raw_eps = float(match.group(1))
            raw_ps = float(match.group(2))
            eps = float(f"{raw_eps:.1e}")
            ps = float(f"{raw_ps:.1e}")
            
            # ★ここで堅牢版関数を呼び出して計算する
            res = calculate_sci_and_jitter_robust(csv_filepath)
            
            if res is not None:
                results.append({
                    'epsilon': eps, 
                    'ps': ps, 
                    'jitter(%)': res['jitter'],
                    'f_AFF(Hz)': res['f_AFF'],
                    'f_MSF(Hz)': res['f_MSF'],
                    'SCI': res['SCI'],
                    'status': res['status']
                })

    # --- データ保存 ---
    if not results:
        print("  [エラー] 有効なデータが見つかりませんでした。スキップします。")
        continue

    df_results = pd.DataFrame(results)
    df_results = df_results.drop_duplicates(subset=['epsilon', 'ps'], keep='last')
    df_results = df_results.sort_values(by=['epsilon', 'ps'])

    print(f"  データを {output_csv} に保存中...")
    df_results.to_csv(output_csv, index=False)
    
print("\n全セットのSCI解析が完了しました.")