# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import glob
import os

# --- ここで設定を変更できます ---
# C++がCSVファイルを出力したフォルダ
INPUT_FOLDER = "simulation_results_1_x0=0.02/"

# 生成されたスペクトログラム画像を保存するフォルダ
OUTPUT_FOLDER = "spectrogram_images_1_x0=0.02/"

# スペクトログラムの解像度 (ベースファイルの設定を維持)
nperseg = 245760
noverlap = 184320
window_type = 'blackmanharris'

# カラースケール設定 (ベースファイルの設定を維持)
cmap = 'gray_r'
vmax = -38.0
vmin = -38.002
# --------------------------------

# --- 出力フォルダが存在しない場合に作成 ---
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"作成しました: {OUTPUT_FOLDER}")
# ------------------------------------

def generate_spectrogram(csv_filepath, output_image_path):
    """ 1つのCSVファイルからスペクトログラム画像を生成する関数 """
    try:
        # 1. CSVファイルを読み込む
        df = pd.read_csv(csv_filepath)

        if df.empty or 'time' not in df or 'pi' not in df:
            print(f"  [警告] スキップ: {csv_filepath} は空か、必要な列がありません。")
            return

        time = df['time'].values
        pi = df['pi'].values
        total_time = time[-1]

        # 2. サンプリング周波数を計算
        sampling_rate = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1.0

        # 3. 信号の末尾にパディングを追加
        padding_length = nperseg 
        pi_padded = np.pad(pi, (0, padding_length), 'constant')

        # 4. ソノグラムを計算
        f, t, Sxx = spectrogram(pi_padded, fs=sampling_rate, window=window_type, nperseg=nperseg, noverlap=noverlap)

        # 5. 結果を元のシミュレーション時間に合わせて切り取る
        valid_indices = np.where(t <= total_time)[0]
        if len(valid_indices) > 2:
            valid_indices = valid_indices[:-2] # 境界効果を除去
        
        t = t[valid_indices]
        Sxx = Sxx[:, valid_indices]

        # 6. パワーを正規化し、デシベル(dB)に変換
        Sxx_normalized = Sxx / np.max(Sxx) if np.max(Sxx) > 0 else Sxx
        db_Sxx = 10 * np.log10(Sxx_normalized + 1e-10)

        # 7. プロット
        plt.figure(figsize=(10, 10))
        plt.pcolormesh(t, f, db_Sxx, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)

        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.ylim(0, 10000)
        
        # タイトルに元のファイル名の一部を表示
        title_filename = os.path.basename(csv_filepath).replace('.csv', '')
        plt.title(f'Spectrogram of (pi) {title_filename}')
        
        # plt.colorbar(label='Intensity [dB]') # ベースファイルに合わせてコメントアウト

        # 8. グラフを画像ファイルとして保存
        plt.savefig(output_image_path)
        plt.close() # メモリを解放するために図を閉じる

    except Exception as e:
        print(f"  [エラー] {csv_filepath} の処理中にエラーが発生しました: {e}")
        plt.close() # エラー時も図を閉じる

# --- メイン処理 ---
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
    generate_spectrogram(csv_filepath, output_image_path)

print(f"\nすべての処理が完了しました。画像は '{OUTPUT_FOLDER}' に保存されました。")