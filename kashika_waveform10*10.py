# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# =========================================================
# --- ここで複数のフォルダペアを設定します ---
FOLDER_PAIRS = [
    ("simulation_results_1_f0=4.9e4_x0=0.02_low epsilon/", "waveform_1_f0=4.9e4_x0=0.02_low epsilon/"),
    #("simulation_results_1_f0=0.4e7_x0=0.02_low epsilon/", "waveform_1_f0=0.4e7_x0=0.02_low epsilon/"),
    #("simulation_results_1_f0=0.7e7_x0=0.02_low epsilon/", "waveform_1_f0=0.7e7_x0=0.02_low epsilon/"),
    ("simulation_results_1_f0=4.9e4_x0=0.02/", "waveform_1_f0=4.9e4_x0=0.02/"),
    #("simulation_results_1_f0=0.4e7_x0=0.02/", "waveform_1_f0=0.4e7_x0=0.02/"),
    #("simulation_results_1_f0=0.7e7_x0=0.02/", "waveform_1_f0=0.7e7_x0=0.02/"),
    # 必要に応じてここに行を追加してください
]

# グラフで表示する時間の範囲指定 (全フォルダ共通)
time_start = 0.0  # 開始時間
time_end = 0.05   # 終了時間
# =========================================================

def generate_waveform(csv_filepath, output_image_path):
    """ 1つのCSVファイルから波形画像を生成する関数 """
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(csv_filepath)

        if df.empty or 'time' not in df or 'pi' not in df:
             # x_left, x_rightのチェックも必要なら追加してください
             print(f"  [警告] スキップ: {csv_filepath} は空か、必要な列がありません。")
             return

        time = df['time'].values
        # カラム名が存在するか確認してから取得（エラー回避のため）
        if 'x_left' in df.columns and 'x_right' in df.columns:
            x_left = df['x_left'].values
            x_right = df['x_right'].values
        else:
            print(f"  [警告] スキップ: {csv_filepath} に 'x_left' または 'x_right' がありません。")
            return

        # グラフプロット(並べて）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # 上グラフ(左音源)
        ax1.plot(time, x_left, color='royalblue')
        ax1.set_title('Left Source Waveform')
        ax1.set_ylabel('Displacement [cm]')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.axhline(0, color='black', linewidth=0.8)

        # 下グラフ(右音源)
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

# --- メイン処理 ---
print("一括処理を開始します...\n")

# 設定されたフォルダペアの数だけループする
for batch_idx, (input_folder, output_folder) in enumerate(FOLDER_PAIRS):
    print(f"==========================================")
    print(f"フォルダセット {batch_idx + 1}/{len(FOLDER_PAIRS)} を処理中")
    print(f"  入力: {input_folder}")
    print(f"  出力: {output_folder}")
    print(f"==========================================")

    # 入力フォルダの存在確認
    if not os.path.exists(input_folder):
        print(f"  [スキップ] 入力フォルダが見つかりません: {input_folder}")
        continue

    # 出力フォルダが存在しない場合に作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"  作成しました: {output_folder}")

    # 入力フォルダ内のすべてのCSVファイルのリストを取得
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    if not csv_files:
        print(f"  [警告] '{input_folder}' にCSVファイルが見つかりません。")
        continue

    print(f"  -> {len(csv_files)} 個のファイルが見つかりました。画像生成を開始します。")

    # 各CSVファイルに対してループ処理
    for i, csv_filepath in enumerate(csv_files):
        # 出力する画像ファイル名を決定
        output_filename = os.path.basename(csv_filepath).replace('.csv', '.png')
        output_image_path = os.path.join(output_folder, output_filename)
        
        # 進捗表示 (例: 10件ごと)
        if (i+1) % 10 == 0 or (i+1) == len(csv_files):
            print(f"    処理中 ({i+1}/{len(csv_files)}): ...{output_filename}")
        
        # 波形画像を生成
        generate_waveform(csv_filepath, output_image_path)

    print(f"  -> 完了: {output_folder}\n")

print("すべてのフォルダセットの処理が完了しました。")