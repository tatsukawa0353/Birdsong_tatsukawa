# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import matplotlib
matplotlib.use('Agg') # 画像保存用バックエンド

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import glob
import os

# --- 設定項目 ---
# 読み込むデータがあるフォルダ
INPUT_FOLDER = "simulation_results_1_x0=0.02/"

# グラフ画像を保存するフォルダ
OUTPUT_FOLDER = "psd_graphs_1_x0=0.02_SPL_v2/"

# スペクトログラム生成パラメータ
nperseg = 245760
noverlap = 184320
window_type = 'blackmanharris'

# グラフの表示範囲
FREQ_MAX = 10000  # Hz
# ----------------

# ★物理定数の設定（基準値をシミュレーション単位へ変換）★
P0_PA = 20e-6          # 現実の基準音圧: 20 μPa
UNIT_CONV = 0.0005     # シミュレーション値をPaに直す係数 (M*0.1)

# シミュレーション単位での基準値 (約 0.04 になります)
# 「現実で20μPaになるのは、シミュレーションでいうといくらか？」を計算
P0_SIM = P0_PA / UNIT_CONV 

# 出力フォルダ作成
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"保存先フォルダを作成しました: {OUTPUT_FOLDER}")
    print(f"基準値設定: p0(sim) = {P0_SIM} (これを二乗して割ります)")

def plot_spl_spectrum_v2(csv_filepath, output_image_path):
    try:
        # 1. データ読み込み
        df = pd.read_csv(csv_filepath)
        if df.empty or 'pi' not in df: return

        time = df['time'].values
        pi = df['pi'].values
        
        if len(time) > 1:
            dt = time[1] - time[0]
            fs = 1.0 / dt
        else:
            return

        # 2. パディング処理
        padding_length = nperseg
        pi_padded = np.pad(pi, (0, padding_length), 'constant')

        # 3. スペクトログラム計算 (出力は pi^2 相当のパワー)
        f, t, Sxx = spectrogram(pi_padded, fs=fs, window=window_type, nperseg=nperseg, noverlap=noverlap)

        # 4. 時間平均をとる
        mean_spectrum = np.mean(Sxx, axis=1)

        # 5. dB SPL への変換 (ご指定の方法)
        # mean_spectrum はそのまま（シミュレーション単位の二乗）
        # それを「シミュレーション単位に換算した基準値の二乗」で割る
        spl_values = 10 * np.log10(mean_spectrum / (P0_SIM ** 2) + 1e-10)

        # 6. グラフプロット
        plt.figure(figsize=(10, 6))
        plt.plot(f, spl_values, linewidth=1.5, color='blue')
        
        plt.title(f"Time-Averaged Power Spectrum (SPL)\n{os.path.basename(csv_filepath)}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Sound Pressure Level [dB SPL]")
        
        plt.xlim(0, FREQ_MAX)
        # 必要に応じてy軸範囲を固定
        # plt.ylim(-20, 120) 
        plt.grid(True, which="both", ls="-", alpha=0.5)

        plt.savefig(output_image_path)
        plt.close()
        
        print(f"保存完了: {output_image_path}")

    except Exception as e:
        print(f"エラー発生 ({csv_filepath}): {e}")

# --- メイン処理 ---
csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))

if not csv_files:
    print(f"エラー: {INPUT_FOLDER} にCSVファイルが見つかりません。")
else:
    print(f"{len(csv_files)} 個のファイルを処理します...")
    for i, csv_file in enumerate(csv_files):
        file_name = os.path.basename(csv_file).replace('.csv', '.png')
        output_path = os.path.join(OUTPUT_FOLDER, file_name)
        
        plot_spl_spectrum_v2(csv_file, output_path)

    print("すべての処理が完了しました。")