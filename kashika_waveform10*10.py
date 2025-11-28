# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import glob
import os

#グラフで表示する時間の範囲指定
time_start = 0.0  # 開始時間
time_end = 0.05   # 終了時間

# C++がCSVファイルを出力したフォルダ
INPUT_FOLDER = "simulation_results_1_x0=0.02/"

# 生成されたスペクトログラム画像を保存するフォルダ
OUTPUT_FOLDER = "waveform_1_x0=0.02/"

# --- 出力フォルダが存在しない場合に作成 ---
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"作成しました: {OUTPUT_FOLDER}")

def generate_waveform(csv_filepath, output_image_path):
    
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(csv_filepath)

        if df.empty or 'time' not in df or 'pi' not in df:
         print(f"  [警告] スキップ: {csv_filepath} は空か、必要な列がありません。")
         return

        time = df['time'].values
        x_left = df['x_left'].values
        x_right = df['x_right'].values

       # グラフプロット(並べて）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

       #上グラフ(左音源)
        ax1.plot(time, x_left, color='royalblue')
        ax1.set_title('Left Source Waveform')
        ax1.set_ylabel('Displacement [cm]')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.axhline(0, color='black', linewidth=0.8)

       #下グラフ(右音源)
        ax2.plot(time, x_right, color='seagreen')
        ax2.set_title('Right Source Waveform')
        ax2.set_ylabel('Displacement [cm]')
        ax2.set_xlabel('Time [sec]')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.axhline(0, color='black', linewidth=0.8)

       # --- X軸の表示範囲を指定 ---
        ax2.set_xlim([time_start, time_end])

       # レイアウトを整える
        plt.tight_layout()

       # グラフを画像ファイルとして保存
        plt.savefig(output_image_path)
        plt.close()

    except Exception as e:
       print(f"  [エラー] {csv_filepath} の処理中にエラーが発生しました: {e}")
       plt.close() # エラー時も図を閉じる

#メイン処理
print(f"'{INPUT_FOLDER}' 内のCSVファイルの処理を開始します...")

# 入力フォルダ内のすべてのCSVファイルのリストを取得
csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))

if not csv_files:
    print(f"エラー: '{INPUT_FOLDER}' にCSVファイルが見つかりません。")
    print("C++のシミュレーションを先に実行してください。")
    exit()

print(f"{len(csv_files)} 個のファイルが見つかりました。")

# 各CSVファイルに対してループ処理
for i, csv_filepath in enumerate(csv_files):
    # 出力する画像ファイル名を決定
    # 例: sim_output_eps_..._ps_....csv -> sim_output_eps_..._ps_....png
    output_filename = os.path.basename(csv_filepath).replace('.csv', '.png')
    output_image_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    print(f"({i+1}/{len(csv_files)}) 処理中: {csv_filepath} -> {output_image_path}")
    
    # スペクトログラムを生成
    generate_waveform(csv_filepath, output_image_path)

print(f"\nすべての処理が完了しました。画像は '{OUTPUT_FOLDER}' に保存されました。")