import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import glob
import os

# 2. スタイル適用（既存の設定より前に書くのがポイント）
plt.style.use(['science', 'ieee'])

# =========================================================
# --- ここで複数のフォルダペアを設定します ---
FOLDER_PAIRS = [
    #("simulation_results_1_f0=1.0e7_x0=0.02_sr(t)_low epsilon/", "waveform_sr_1_f0=1.0e7_low epsilon/"),
    #("simulation_results_1_f0=1.0e7_x0=0.02_sr(t)/", "waveform_sr_1_f0=1.0e7/"),
    ("simulation_results_1_f0=0.1e7_x0=0.02_sr(t)_low/", "waveform_sr_1_f0=0.1e7_low/")
]

# 物理定数とスケーリング
M = 5.0e-3
P_SCALE = M / 10000 

# グラフで表示する時間の範囲指定 [sec]
time_start = 0.0
time_end = 0.05
# =========================================================

def generate_waveform(csv_filepath, output_image_path):
    """ 1つのCSVファイルから s_l と s_r の正方形波形画像を生成する関数 """
    try:
        df = pd.read_csv(csv_filepath)

        required_cols = ['time', 's_l', 's_r']
        if not all(col in df.columns for col in required_cols):
            print(f"  [警告] スキップ: {csv_filepath} に必要な列がありません。")
            return

        time = df['time'].values
        sl_kpa = df['s_l'].values * P_SCALE
        sr_kpa = df['s_r'].values * P_SCALE

        # --- 【変更】figsizeを(12, 12)にして正方形にする ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        
        # --- フォントサイズの設定 ---
        label_size = 46  
        tick_size = 38   # 数字が重ならないよう少しだけ小さく設定

        # 上グラフ (左音源 s_l)
        ax1.plot(time, sl_kpa, color='royalblue', label='$s_l$')
        # ラベルが長すぎると画像からはみ出るため、短縮
        ax1.set_ylabel('$s_l$ [kPa]', fontsize=label_size)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.tick_params(axis='y', labelsize=tick_size)
        #ax1.legend(loc='upper right', fontsize=24) # レジェンドも少し大きく

        # 下グラフ (右圧力擾乱 s_r)
        ax2.plot(time, sr_kpa, color='crimson', label='$s_r$')
        ax2.set_ylabel('$s_r$ [kPa]', fontsize=label_size)
        ax2.set_xlabel('Time [s]', fontsize=label_size)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.axhline(0, color='black', linewidth=0.8)
        
        # 指定されていた y 軸範囲の設定
        ax2.set_ylim(-0.12, 0)
        
        # --- X軸と全体の目盛り設定 ---
        ax2.set_xlim([time_start, time_end])
        ax2.tick_params(axis='both', which='major', labelsize=tick_size)
        #ax2.legend(loc='upper right', fontsize=24)

        # --- 全ての軸（ax1, ax2）の枠と目盛りを統一設定 ---
        for ax in [ax1, ax2]:
            # 1. 枠線の太さと色を黒に固定
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2.0)
            
            # 2. 目盛りの設定（黒、太さ、内向き、上・右なし）
            ax.tick_params(
                axis='both', 
                which='major', 
                labelsize=tick_size, 
                colors='black', 
                width=2.0, 
                length=10, 
                direction='in', 
                top=False, 
                right=False
            )

        # レイアウトを整える
        plt.tight_layout()

        fig.align_ylabels([ax1, ax2])

        # --- 【変更】余白を削って保存 ---
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close()

    except Exception as e:
        print(f"  [エラー] {csv_filepath} の処理中にエラーが発生しました: {e}")
        plt.close()

# --- メイン処理（変更なし） ---
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