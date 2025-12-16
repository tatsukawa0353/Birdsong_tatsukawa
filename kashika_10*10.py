# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import glob
import os

# =========================================================
# --- ここで複数のフォルダペアを設定します ---
# リストの中に ("入力フォルダパス", "出力フォルダパス"), の形で記述してください
FOLDER_PAIRS = [
    ("simulation_results_1_f0=0.1e7_x0=0.02_low epsilon/", "spectrogram_images_1_f0=0.1e7_x0=0.02_low epsilon/"),
    ("simulation_results_1_f0=0.4e7_x0=0.02_low epsilon/", "spectrogram_images_1_f0=0.4e7_x0=0.02_low epsilon/"),
    ("simulation_results_1_f0=0.7e7_x0=0.02_low epsilon/", "spectrogram_images_1_f0=0.7e7_x0=0.02_low epsilon/"),
    ("simulation_results_1_f0=0.1e7_x0=0.02/", "spectrogram_images_1_f0=0.1e7_x0=0.02"),
    ("simulation_results_1_f0=0.4e7_x0=0.02/", "spectrogram_images_1_f0=0.4e7_x0=0.02"),
    ("simulation_results_1_f0=0.7e7_x0=0.02/", "spectrogram_images_1_f0=0.7e7_x0=0.02"),
    # 必要に応じてここに行を追加してください
]

# スペクトログラムの設定 (全フォルダ共通)
nperseg = 245760
noverlap = 184320
window_type = 'blackmanharris'

# カラースケール設定 (全フォルダ共通)
cmap = 'gray_r'
vmax = -38.0
vmin = -38.002
# =========================================================

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
        plt.title(f'Spectrogram of {title_filename}')
        
        # 8. グラフを画像ファイルとして保存
        plt.savefig(output_image_path)
        plt.close() # メモリを解放するために図を閉じる

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
        continue  # 次のフォルダペアへ進む

    # 出力フォルダが存在しない場合に作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"  作成しました: {output_folder}")

    # 入力フォルダ内のすべてのCSVファイルのリストを取得
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    if not csv_files:
        print(f"  [警告] '{input_folder}' にCSVファイルが見つかりません。")
        continue

    print(f"  -> {len(csv_files)} 個のファイルが見つかりました。変換を開始します。")

    # 各CSVファイルに対してループ処理
    for i, csv_filepath in enumerate(csv_files):
        # 出力パスの生成
        output_filename = os.path.basename(csv_filepath).replace('.csv', '.png')
        output_image_path = os.path.join(output_folder, output_filename)
        
        # 進捗表示 (例: 1/100)
        if (i+1) % 10 == 0 or (i+1) == len(csv_files): # ログが多すぎないように10件ごとまたは最後に表示
             print(f"    処理中 ({i+1}/{len(csv_files)}): ...{output_filename}")
        
        # スペクトログラムを生成
        generate_spectrogram(csv_filepath, output_image_path)
    
    print(f"  -> 完了: {output_folder}\n")

print("すべてのフォルダセットの処理が完了しました。")