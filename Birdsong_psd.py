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
INPUT_FOLDER = "simulation_results_2_x0=0.02_low parameters epsilon/"

# グラフ画像を保存するフォルダ
OUTPUT_FOLDER = "psd_graphs_2_x0=0.02_low parameters epsilon normal/"

# スペクトログラム生成パラメータ (以前の設定と合わせる)
nperseg = 245760
noverlap = 184320
window_type = 'blackmanharris'

# グラフの表示範囲
FREQ_MAX = 10000  # Hz (10kHzまで表示)
# ----------------

# 出力フォルダ作成
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"保存先フォルダを作成しました: {OUTPUT_FOLDER}")

def plot_time_averaged_spectrum(csv_filepath, output_image_path):
    try:
        # 1. データ読み込み
        df = pd.read_csv(csv_filepath)
        if df.empty or 'pi' not in df:
            print(f"スキップ: {csv_filepath} (データなし)")
            return

        time = df['time'].values
        pi = df['pi'].values
        
        # サンプリングレート計算
        if len(time) > 1:
            dt = time[1] - time[0]
            fs = 1.0 / dt
        else:
            return

        # 2. パディング処理 (npersegが大きい場合の対策)
        padding_length = nperseg
        pi_padded = np.pad(pi, (0, padding_length), 'constant')

        # 3. スペクトログラム計算
        f, t, Sxx = spectrogram(pi_padded, fs=fs, window=window_type, nperseg=nperseg, noverlap=noverlap)

        # 4. 時間方向(axis=1)に対して平均をとる = 時間平均パワースペクトル
        mean_spectrum = np.mean(Sxx, axis=1)

        # 5. dB変換 (対数スケール)
        # log10(0)を防ぐため微小値を加える
        mean_spectrum_db = 10 * np.log10(mean_spectrum + 1e-10)

        # 6. グラフプロット
        plt.figure(figsize=(10, 6))
        plt.plot(f, mean_spectrum_db, linewidth=1.5, color='blue')
        
        plt.title(f"Time-Averaged Power Spectrum\n{os.path.basename(csv_filepath)}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power Spectral Density [dB]")
        plt.xlim(0, FREQ_MAX)  # 周波数範囲を制限
        plt.grid(True, which="both", ls="-", alpha=0.5)

        # 画像保存
        plt.savefig(output_image_path)
        plt.close() # メモリ解放
        
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
        # 出力ファイル名を作成 (.csv -> .png)
        file_name = os.path.basename(csv_file).replace('.csv', '.png')
        output_path = os.path.join(OUTPUT_FOLDER, file_name)
        
        plot_time_averaged_spectrum(csv_file, output_path)

    print("すべての処理が完了しました。")