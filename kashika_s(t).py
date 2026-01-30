import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# =========================================================
# --- ここで複数のフォルダペアを設定します ---
FOLDER_PAIRS = [
    ("simulation_results_1_f0=1.0e7_x0=0.02_sr(t)_low epsilon/", "waveform_sr_1_f0=1.0e7_low epsilon/"),
    ("simulation_results_1_f0=1.0e7_x0=0.02_sr(t)/", "waveform_sr_1_f0=1.0e7/"),
]

# 物理定数とスケーリング (cm/s^2 -> kPa)
M = 5.0e-3
P_SCALE = M / 10000 

# グラフで表示する時間の範囲指定 [sec]
# 波形をしっかり見るため、定常状態の後半（例：0.1〜0.12s）を推奨
time_start = 0.0
time_end = 0.12
# =========================================================

def generate_waveform(csv_filepath, output_image_path):
    """ 1つのCSVファイルから s_l と s_r の波形画像を生成する関数 """
    try:
        df = pd.read_csv(csv_filepath)

        # 必要なカラムの存在チェック
        required_cols = ['time', 's_l', 's_r']
        if not all(col in df.columns for col in required_cols):
            print(f"  [警告] スキップ: {csv_filepath} に必要な列がありません。")
            return

        # データの抽出と単位変換
        time = df['time'].values
        sl_kpa = df['s_l'].values * P_SCALE
        sr_kpa = df['s_r'].values * P_SCALE

        # グラフプロット (2段)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # 上グラフ (左音源 s_l)
        ax1.plot(time, sl_kpa, color='royalblue', label='Left Source $s_l$')
        ax1.set_title(f'Waveform Analysis: {os.path.basename(csv_filepath)}')
        ax1.set_ylabel('Pressure $s_l$ [kPa]')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper right')

        # 下グラフ (右圧力擾乱 s_r)
        ax2.plot(time, sr_kpa, color='crimson', label='Right Perturbation $s_r$')
        ax2.set_ylabel('Pressure $s_r$ [kPa]')
        ax2.set_xlabel('Time [sec]')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='upper right')

        # --- X軸の表示範囲を指定 ---
        ax2.set_xlim([time_start, time_end])

        plt.tight_layout()
        plt.savefig(output_image_path, dpi=150)
        plt.close()

    except Exception as e:
        print(f"  [エラー] {csv_filepath} の処理中にエラーが発生しました: {e}")
        plt.close()

# --- メイン処理 ---
print("一括処理を開始します...\n")

for batch_idx, (input_folder, output_folder) in enumerate(FOLDER_PAIRS):
    print(f"==========================================")
    print(f"フォルダセット {batch_idx + 1}/{len(FOLDER_PAIRS)} を処理中")
    
    if not os.path.exists(input_folder):
        print(f"  [スキップ] 入力フォルダが見つかりません: {input_folder}")
        continue
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        print(f"  [警告] '{input_folder}' にCSVが見つかりません。")
        continue

    for i, csv_filepath in enumerate(sorted(csv_files)):
        output_filename = os.path.basename(csv_filepath).replace('.csv', '.png')
        output_image_path = os.path.join(output_folder, output_filename)
        
        if (i+1) % 10 == 0 or (i+1) == len(csv_files):
            print(f"    処理中 ({i+1}/{len(csv_files)}): {output_filename}")
        
        generate_waveform(csv_filepath, output_image_path)

print("\nすべての処理が完了しました。")