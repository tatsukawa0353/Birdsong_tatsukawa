import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.signal import correlate, spectrogram, welch
from scipy.stats import entropy


# ==========================================
# 設定エリア
# ==========================================

ANALYSIS_SETS = [
    #{
        #"input_folders": [
        #    "simulation_results_1_x0=0.02_low parameters epsilon/",
       #     "simulation_results_1_x0=0.02/",
      #  ],
     #   "output_csv": "hnr_sci_aci_se_1_f0=1.0e7_3.4.csv",
    #},
    {
        "input_folders": [
            "simulation_results_1_f0=1.0e7_x0=0.02_update/",
        ],
        "output_csv": "hnr_sci_aci_se_1_f0=1.0e7_update_2.csv",
    },
    
]

# SCI / HNR
SCI_HNR_START_TIME = 0.0
CALC_MIN_FREQ = 250.0
CALC_MAX_FREQ = 12000.0
F0_MAX_FREQ = 3500.0
HNR_THRESHOLD_DB = 3.0
HNR_SILENCE_THRESHOLD = 1e5
HNR_FRAME_SEC = 0.01
HNR_HOP_SEC = 0.005
MIN_ACTIVE_RATIO = 0.7
MIN_ACTIVE_DURATION = 0.08
F0_SWEEP_IQR_RATIO = 0.05
HNR_SWEEP_PERCENTILE = 75
F0_BOUNDARY_MARGIN = 0.98
USE_SPECTRAL_TONALITY_HNR = False

# ACI
ACI_START_TIME = 0.0
ACI_NPERSEG = 245760
ACI_NOVERLAP = 184320
ACI_WINDOW = "blackmanharris"
ACI_NOISE_THRESHOLD_RATIO = 0.01
ACI_MIN_POWER_THRESHOLD = 1e5

# SE (Spectral Entropy): Birdsong_complex_ver2.0.py と同じ2段階チェック方式
SE_START_TIME = 0.0
SE_NPERSEG = 245760
SE_NOVERLAP = 184320
SE_WINDOW = "blackmanharris"
SE_CHECK_MIN_FREQ = 250.0
SE_CHECK_MAX_FREQ = 12000.0
SE_CALC_MIN_FREQ = 250.0
SE_CALC_MAX_FREQ = 12000.0
SE_SILENCE_THRESHOLD = 1e5

FILENAME_PATTERN = re.compile(
    r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv"
)


# ==========================================
# 共通処理
# ==========================================

def get_sampling_rate(df):
    time_arr = df["time"].values
    if len(time_arr) < 2:
        return None
    dt = time_arr[1] - time_arr[0]
    if dt <= 0:
        return None
    return 1.0 / dt


def get_signal_segment(df, start_time):
    if df.empty or "pi" not in df or "time" not in df:
        return None

    df_segment = df[df["time"] >= start_time]
    if len(df_segment) < 2:
        return None

    pi = df_segment["pi"].values
    if np.isnan(pi).any() or np.isinf(pi).any():
        return None

    return pi - np.mean(pi)


def detect_f0_and_hnr_by_frame(pi_norm, fs):
    """
    フレームごとにF0とHNRを計算し、全体の代表値（中央値）を返す
    """
    frame_len = int(HNR_FRAME_SEC * fs)
    hop_len = int(HNR_HOP_SEC * fs)
    min_lag = max(1, int(fs / F0_MAX_FREQ))
    max_lag = int(fs / CALC_MIN_FREQ)

    if frame_len <= max_lag or hop_len <= 0 or len(pi_norm) < frame_len:
        return np.nan, np.nan

    f0_candidates = []
    hnr_candidates = []

    for i in range(0, len(pi_norm) - frame_len, hop_len):
        frame = pi_norm[i : i + frame_len]
        frame = frame - np.mean(frame)
        frame_energy = np.sum(frame * frame)
        
        if frame_energy <= 0:
            continue

        corr = correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :] # 自己相関関数
        
        if len(corr) <= max_lag:
            continue

        valid_corr = corr[min_lag:max_lag]
        if valid_corr.size == 0:
            continue

        # フレーム内の最も強い周期（ラグ）を探す
        best_lag_idx = int(np.argmax(valid_corr))
        best_lag = min_lag + best_lag_idx
        
        # フレームごとのHNRを計算
        r = valid_corr[best_lag_idx] / corr[0]
        
        if r > 0 and r < 1.0:
            frame_hnr_db = 10.0 * np.log10(r / (1.0 - r))
            frame_f0 = fs / best_lag
            
            hnr_candidates.append(frame_hnr_db)
            f0_candidates.append(frame_f0)

    if not hnr_candidates:
        return np.nan, np.nan

    f0_candidates = np.array(f0_candidates)
    hnr_candidates = np.array(hnr_candidates)

    # 全フレームの中央値をその音の代表F0とする。
    # F0が時間変化している音は、前半のノイズ区間に中央値HNRが引っ張られやすいので
    # HNRだけ上位側の分位点も候補にする。
    true_f0 = float(np.median(f0_candidates))
    hnr_median = float(np.median(hnr_candidates))
    f0_iqr_ratio = (
        (np.percentile(f0_candidates, 75) - np.percentile(f0_candidates, 25)) / true_f0
        if true_f0 > 0
        else 0.0
    )

    if f0_iqr_ratio >= F0_SWEEP_IQR_RATIO:
        overall_hnr = max(
            hnr_median,
            float(np.percentile(hnr_candidates, HNR_SWEEP_PERCENTILE)),
        )
    else:
        overall_hnr = hnr_median
    
    return true_f0, overall_hnr

# ==========================================

def calculate_spectral_tonality_hnr(pi_norm, fs):
    frame_len = int(0.02 * fs)
    hop_len = int(0.01 * fs)
    if frame_len < 8 or hop_len <= 0 or len(pi_norm) < frame_len:
        return np.nan

    noverlap = max(0, frame_len - hop_len)
    f, _, sxx = spectrogram(
        pi_norm,
        fs=fs,
        window="blackmanharris",
        nperseg=frame_len,
        noverlap=noverlap,
    )

    valid_indices = (f >= CALC_MIN_FREQ) & (f <= CALC_MAX_FREQ)
    f_valid = f[valid_indices]
    sxx_valid = sxx[valid_indices, :]
    if f_valid.size == 0 or sxx_valid.size == 0:
        return np.nan

    frame_hnr_values = []
    peak_half_width_hz = 150.0

    for frame_power in sxx_valid.T:
        total_power = np.sum(frame_power)
        if total_power <= 0:
            continue

        peak_idx = int(np.argmax(frame_power))
        peak_freq = f_valid[peak_idx]
        peak_band = np.abs(f_valid - peak_freq) <= peak_half_width_hz
        harmonic_power = np.sum(frame_power[peak_band])
        noise_power = total_power - harmonic_power

        if harmonic_power <= 0 or noise_power <= 0:
            continue

        frame_hnr_values.append(10.0 * np.log10(harmonic_power / noise_power))

    if not frame_hnr_values:
        return np.nan

    return float(np.median(frame_hnr_values))


def has_enough_band_power(pi, fs):
    nperseg = min(245760, len(pi))
    if nperseg < 8:
        return False

    noverlap = min(184320, nperseg - 1)
    f, _, sxx = spectrogram(
        pi,
        fs=fs,
        window="blackmanharris",
        nperseg=nperseg,
        noverlap=noverlap,
    )

    valid_indices = (f >= CALC_MIN_FREQ) & (f <= CALC_MAX_FREQ)
    if not np.any(valid_indices):
        return False

    mean_spectrum = np.mean(sxx[valid_indices, :], axis=1)
    return np.max(mean_spectrum) >= HNR_SILENCE_THRESHOLD


def has_sustained_band_power(pi, fs):
    frame_len = int(HNR_FRAME_SEC * fs)
    hop_len = int(HNR_HOP_SEC * fs)
    if frame_len < 8 or hop_len <= 0 or len(pi) < frame_len:
        return False

    noverlap = max(0, frame_len - hop_len)
    f, _, sxx = spectrogram(
        pi,
        fs=fs,
        window="blackmanharris",
        nperseg=frame_len,
        noverlap=noverlap,
    )

    valid_indices = (f >= CALC_MIN_FREQ) & (f <= CALC_MAX_FREQ)
    sxx_valid = sxx[valid_indices, :]
    if sxx_valid.size == 0:
        return False

    frame_peak_power = np.max(sxx_valid, axis=0)
    active_frames = frame_peak_power >= HNR_SILENCE_THRESHOLD
    active_ratio = np.mean(active_frames)
    active_duration = np.sum(active_frames) * HNR_HOP_SEC

    return active_ratio >= MIN_ACTIVE_RATIO and active_duration >= MIN_ACTIVE_DURATION


def calculate_sci_with_hnr_filter(df):
    fs = get_sampling_rate(df)
    if fs is None:
        return {"HNR(dB)": np.nan, "f_AFF(Hz)": np.nan, "f_MSF(Hz)": np.nan, "SCI": np.nan, "status": "Invalid_Time"}

    pi_norm = get_signal_segment(df, SCI_HNR_START_TIME)
    if pi_norm is None:
        return {"HNR(dB)": np.nan, "f_AFF(Hz)": np.nan, "f_MSF(Hz)": np.nan, "SCI": np.nan, "status": "Invalid_or_Diverged"}

    if len(pi_norm) < 1000:
        return {"HNR(dB)": np.nan, "f_AFF(Hz)": np.nan, "f_MSF(Hz)": np.nan, "SCI": np.nan, "status": "Too_Short"}

    if not has_enough_band_power(pi_norm, fs):
        return {"HNR(dB)": np.nan, "f_AFF(Hz)": np.nan, "f_MSF(Hz)": np.nan, "SCI": np.nan, "status": "No_Peaks"}

    if not has_sustained_band_power(pi_norm, fs):
        return {"HNR(dB)": np.nan, "f_AFF(Hz)": np.nan, "f_MSF(Hz)": np.nan, "SCI": np.nan, "status": "Short_Active"}

    # ★ ここで新しいフレーム単位の計算関数を呼び出す
    true_f0, hnr_db = detect_f0_and_hnr_by_frame(pi_norm, fs)
    
    if np.isnan(true_f0) or true_f0 <= 0:
        return {"HNR(dB)": np.nan, "f_AFF(Hz)": np.nan, "f_MSF(Hz)": np.nan, "SCI": np.nan, "status": "F0_Detection_Failed"}

    if true_f0 >= F0_MAX_FREQ * F0_BOUNDARY_MARGIN:
        return {"HNR(dB)": hnr_db, "f_AFF(Hz)": true_f0, "f_MSF(Hz)": np.nan, "SCI": np.nan, "status": "F0_Boundary"}

    if USE_SPECTRAL_TONALITY_HNR:
        spectral_hnr_db = calculate_spectral_tonality_hnr(pi_norm, fs)
        if not np.isnan(spectral_hnr_db):
            hnr_db = max(hnr_db, spectral_hnr_db)

    # ★ HNRの閾値判定
    if np.isnan(hnr_db) or hnr_db < HNR_THRESHOLD_DB:
        return {"HNR(dB)": hnr_db, "f_AFF(Hz)": true_f0, "f_MSF(Hz)": np.nan, "SCI": np.nan, "status": "Low_HNR"}

    # --- 以降は元のSCI計算と同じ ---
    nperseg = min(245760, len(pi_norm))
    if nperseg < 8:
        return {"HNR(dB)": hnr_db, "f_AFF(Hz)": true_f0, "f_MSF(Hz)": np.nan, "SCI": np.nan, "status": "Too_Short_For_SCI"}

    noverlap = min(184320, nperseg - 1)
    f, pxx = welch(pi_norm, fs=fs, nperseg=nperseg, noverlap=noverlap, window="blackmanharris")

    valid_indices = (f >= CALC_MIN_FREQ) & (f <= CALC_MAX_FREQ)
    f_valid = f[valid_indices]
    pxx_valid = pxx[valid_indices]
    total_power = np.sum(pxx_valid)

    if f_valid.size == 0 or total_power <= 0:
        return {"HNR(dB)": hnr_db, "f_AFF(Hz)": true_f0, "f_MSF(Hz)": np.nan, "SCI": np.nan, "status": "No_Power"}

    f_msf = np.sum(f_valid * pxx_valid) / total_power
    sci = f_msf / true_f0

    return {"HNR(dB)": hnr_db, "f_AFF(Hz)": true_f0, "f_MSF(Hz)": f_msf, "SCI": sci, "status": "OK"}


def calculate_aci(df):
    fs = get_sampling_rate(df)
    pi_norm = get_signal_segment(df, ACI_START_TIME)
    if fs is None or pi_norm is None or len(pi_norm) < ACI_NPERSEG:
        return 0.0

    f, _, sxx = spectrogram(
        pi_norm,
        fs=fs,
        window=ACI_WINDOW,
        nperseg=ACI_NPERSEG,
        noverlap=ACI_NOVERLAP,
    )

    valid_indices = (f >= CALC_MIN_FREQ) & (f <= CALC_MAX_FREQ)
    sxx_filtered = sxx[valid_indices, :]
    if sxx_filtered.size == 0:
        return 0.0

    sum_power = np.sum(sxx_filtered, axis=1)
    max_power = np.max(sum_power)
    if max_power < ACI_MIN_POWER_THRESHOLD:
        return 0.0

    power_threshold = max_power * ACI_NOISE_THRESHOLD_RATIO
    diff_abs = np.abs(np.diff(sxx_filtered, axis=1))
    sum_diff = np.sum(diff_abs, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        aci_per_bin = sum_diff / sum_power
        aci_per_bin[sum_power < power_threshold] = 0.0
        aci_per_bin = np.nan_to_num(aci_per_bin)

    return float(np.sum(aci_per_bin))


def calculate_spectral_entropy(df):
    fs = get_sampling_rate(df)
    if fs is None or df.empty or "pi" not in df or "time" not in df:
        return 0.0

    df_segment = df[df["time"] >= SE_START_TIME]
    if len(df_segment) < SE_NPERSEG:
        return 0.0

    pi = df_segment["pi"].values
    if np.isnan(pi).any() or np.isinf(pi).any():
        return 0.0

    f, _, sxx = spectrogram(
        pi,
        fs=fs,
        window=SE_WINDOW,
        nperseg=SE_NPERSEG,
        noverlap=SE_NOVERLAP,
    )

    mean_spectrum_full = np.mean(sxx, axis=1)
    check_indices = (f >= SE_CHECK_MIN_FREQ) & (f <= SE_CHECK_MAX_FREQ)
    spectrum_for_check = mean_spectrum_full[check_indices]
    if spectrum_for_check.size == 0:
        return 0.0

    peak_power_in_band = np.max(spectrum_for_check)
    if peak_power_in_band < SE_SILENCE_THRESHOLD:
        return 0.0

    valid_indices = (f >= SE_CALC_MIN_FREQ) & (f <= SE_CALC_MAX_FREQ)
    sxx_filtered = sxx[valid_indices, :]
    if sxx_filtered.size == 0:
        return 0.0

    mean_spectrum = np.mean(sxx_filtered, axis=1)
    total_power = np.sum(mean_spectrum)
    if total_power == 0:
        return 0.0

    psd_norm = mean_spectrum / total_power
    ent = entropy(psd_norm, base=2)
    max_entropy = np.log2(len(psd_norm))
    if max_entropy == 0:
        return 0.0

    return float(ent / max_entropy)


def analyze_file(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath)
        sci_hnr = calculate_sci_with_hnr_filter(df)

        if sci_hnr["status"] != "OK":
            return {
                **sci_hnr,
                "ACI": np.nan,
                "SE": np.nan,
            }

        aci = calculate_aci(df)
        se = calculate_spectral_entropy(df)
        return {
            **sci_hnr,
            "ACI": aci,
            "SE": se,
        }
    except Exception as exc:
        print(f"Error processing {csv_filepath}: {exc}")
        return None


def main():
    print("解析モード: HNRフィルタ付き SCI + ACI + SE")
    print(f"HNRしきい値: {HNR_THRESHOLD_DB} dB 以上でSCIを計算")
    print("3Dマップは作成しません。")

    for set_idx, config in enumerate(ANALYSIS_SETS):
        target_folders = config["input_folders"]
        output_csv = config["output_csv"]

        print("\n" + "=" * 60)
        print(f"解析セット {set_idx + 1} を処理中...")
        print(f"  入力フォルダ数: {len(target_folders)}")
        print(f"  出力CSV: {output_csv}")
        print("=" * 60)

        results = []

        for folder in target_folders:
            if not os.path.exists(folder):
                print(f"  [警告] フォルダが見つかりません: {folder}")
                continue

            csv_files = glob.glob(os.path.join(folder, "*.csv"))
            print(f"  フォルダ読み込み: {folder} ({len(csv_files)} files)")

            for i, csv_filepath in enumerate(csv_files, start=1):
                filename = os.path.basename(csv_filepath)
                match = FILENAME_PATTERN.match(filename)
                if not match:
                    continue

                raw_eps = float(match.group(1))
                raw_ps = float(match.group(2))
                eps = float(f"{raw_eps:.1e}")
                ps = float(f"{raw_ps:.1e}")

                result = analyze_file(csv_filepath)
                if result is None:
                    continue

                results.append(
                    {
                        "epsilon": eps,
                        "ps": ps,
                        "HNR(dB)": result["HNR(dB)"],
                        "f_AFF(Hz)": result["f_AFF(Hz)"],
                        "f_MSF(Hz)": result["f_MSF(Hz)"],
                        "SCI": result["SCI"],
                        "ACI": result["ACI"],
                        "SE": result["SE"],
                        "status": result["status"],
                    }
                )

                if i % 50 == 0:
                    print(f"    Progress: {i} files")

        if not results:
            print("  [エラー] 有効なデータが見つかりませんでした。スキップします。")
            continue

        df_results = pd.DataFrame(results)
        df_results = df_results.drop_duplicates(subset=["epsilon", "ps"], keep="last")
        df_results = df_results.sort_values(by=["epsilon", "ps"])

        print(f"  データを {output_csv} に保存中...")
        df_results.to_csv(output_csv, index=False)

    print("\n全セットのHNR/SCI/ACI/SE解析が完了しました。")


if __name__ == "__main__":
    main()
