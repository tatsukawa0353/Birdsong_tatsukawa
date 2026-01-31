import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================
# --- 設定項目 ---
# =========================================================
# 入力するCSVのパス
INPUT_CSV = "simulation_output_test1.csv"
# 保存する画像の名前
OUTPUT_NAME = "detailed_analysis_single.png"

# 物理定数とスケーリング (cm/s^2 -> kPa)
M = 5.0e-3
P_SCALE = M / 10000 

# グラフで表示する時間の範囲指定 [sec]
time_start = 0.0
time_end = 0.12
# =========================================================

def generate_single_waveform(csv_filepath, output_image_path):
    """ 指定された1つのCSVファイルから波形画像を生成する """
    if not os.path.exists(csv_filepath):
        print(f"[エラー] ファイルが見つかりません: {csv_filepath}")
        return

    try:
        print(f"処理中: {csv_filepath}")
        df = pd.read_csv(csv_filepath)

        # 必要なカラムの存在チェック
        required_cols = ['time', 's_l', 's_r']
        if not all(col in df.columns for col in required_cols):
            print(f"[エラー] 必要な列 ({required_cols}) がCSVに含まれていません。")
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
        print(f"完了: {output_image_path} を保存しました。")

    except Exception as e:
        print(f"[エラー] 処理中にエラーが発生しました: {e}")
        plt.close()

# 実行
if __name__ == "__main__":
    generate_single_waveform(INPUT_CSV, OUTPUT_NAME)