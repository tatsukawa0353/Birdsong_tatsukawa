import pandas as pd
import numpy as np
from scipy.signal import find_peaks, welch
import glob
import os
import re

# ==========================================
# ★★★ 設定エリア ★★★
# ==========================================

# 解析セットの定義（前回のものを引き継ぎ）
ANALYSIS_SETS = [
    # --- セット1 ---
    {
        "input_folders": [
            "simulation_results_1_f0=0.1e7_x0=0.02_low epsilon/",
            "simulation_results_1_f0=0.1e7_x0=0.02/"
        ],
        "output_csv": "sci_data_1_f0=0.1e7.csv",
    },
    # 他のセットも同様に追加してください...
    # (ここでは例として1つだけにしていますが、元のリストをそのままコピペしてOKです)
]

# 分析パラメータ
START_TIME = 0.05          # 解析開始時刻（秒）
CALC_MIN_FREQ = 250.0     # 計算下限周波数（Hz）
CALC_MAX_FREQ = 12000.0   # 計算上限周波数（Hz）
JITTER_THRESHOLD = 5.0    # 許容する最大ジッタ（%）。これより揺らぎが大きい音はノイズとして除外

# ==========================================
# 関数定義（ジッタ計算 + SCI計算）
# ==========================================

def calculate_sci_and_jitter(csv_filepath):
    """ 
    ジッタ(Jitter)を計算して周期性を判定し、
    条件を満たせばSCI (Spectral Content Index) を計算する関数
    """
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty or 'pi' not in df or 'time' not in df:
            return None
        
        # 時間フィルタリング
        df_segment = df[df['time'] >= START_TIME]
        if len(df_segment) < 1000: # データが短すぎる場合は除外
            return None
            
        time_arr = df_segment['time'].values
        pi = df_segment['pi'].values 
        fs = 1.0 / (time_arr[1] - time_arr[0]) # サンプリング周波数

        # ---------------------------------------------------------
        # ★ステップ1: 簡易的なFFTで大まかな基本周波数(F0)を推定
        # ---------------------------------------------------------
        # F0の目安がないと波形のピーク（周期）を正しく探せないため
        fft_val = np.abs(np.fft.rfft(pi))
        freqs = np.fft.rfftfreq(len(pi), 1/fs)
        
        # 指定帯域内での最大ピークを大まかなF0とする
        valid_idx_fft = (freqs >= CALC_MIN_FREQ) & (freqs <= CALC_MAX_FREQ)
        if not np.any(valid_idx_fft):
            return None
            
        f0_guess = freqs[valid_idx_fft][np.argmax(fft_val[valid_idx_fft])]
        expected_period_samples = fs / f0_guess

        # ---------------------------------------------------------
        # ★ステップ2: ジッタ (Jitter) の計算
        # ---------------------------------------------------------
        # 波形のピーク（山）を探す。F0周期の70%以上の間隔を開ける設定
        peaks, _ = find_peaks(pi, distance=expected_period_samples * 0.7)
        
        if len(peaks) < 5:
            # ピークが少なすぎる（＝周期的な音ではない）
            return {'jitter': np.nan, 'f_AFF': np.nan, 'f_MSF': np.nan, 'SCI': np.nan, 'status': 'No_Peaks'}

        # ピーク間のサンプル数（周期 Ti）を計算
        periods = np.diff(peaks) / fs
        
        # Local Jitter (%) の計算: |Ti - Ti+1| の平均 / Tiの平均 * 100
        mean_period = np.mean(periods)
        mean_diff = np.mean(np.abs(np.diff(periods)))
        jitter_percent = (mean_diff / mean_period) * 100.0

        # ジッタが閾値を超えたら「ノイジー（カオス・非周期的）」とみなして除外
        if jitter_percent > JITTER_THRESHOLD:
            return {'jitter': jitter_percent, 'f_AFF': np.nan, 'f_MSF': np.nan, 'SCI': np.nan, 'status': 'High_Jitter'}

        # ---------------------------------------------------------
        # ★ステップ3: SCI (Spectral Content Index) の計算
        # ---------------------------------------------------------
        # ウェルチ法で滑らかなパワースペクトル密度（PSD）を計算
        f, Pxx = welch(pi, fs=fs, nperseg=245760, window='blackmanharris')
        
        # 計算対象の周波数帯を抽出
        valid_indices = (f >= CALC_MIN_FREQ) & (f <= CALC_MAX_FREQ)
        f_valid = f[valid_indices]
        Pxx_valid = Pxx[valid_indices]
        
        if f_valid.size == 0 or np.sum(Pxx_valid) == 0:
            return {'jitter': jitter_percent, 'f_AFF': np.nan, 'f_MSF': np.nan, 'SCI': np.nan, 'status': 'No_Power'}

        # ① 平均基本周波数 (f_AFF): パワースペクトルの最大ピーク周波数
        f_AFF = f_valid[np.argmax(Pxx_valid)]
        
        # ② 平均スペクトル周波数 (f_MSF): スペクトル重心 (Σ(f * P) / ΣP)
        f_MSF = np.sum(f_valid * Pxx_valid) / np.sum(Pxx_valid)
        
        # ③ SCI = f_MSF / f_AFF
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
# メイン処理
# ==========================================

print(f"解析モード: SCI解析 ＆ Jitterフィルタリング")
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
            
            # JitterとSCIの計算
            res = calculate_sci_and_jitter(csv_filepath)
            
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
    # 重複排除とソート
    df_results = df_results.drop_duplicates(subset=['epsilon', 'ps'], keep='last')
    df_results = df_results.sort_values(by=['epsilon', 'ps'])

    print(f"  データを {output_csv} に保存中...")
    df_results.to_csv(output_csv, index=False)
    
print("\n全セットのSCI解析が完了しました。")