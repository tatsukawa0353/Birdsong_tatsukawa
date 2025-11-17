# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# --- 【重要】あなたが手動で作成したCSVファイルの名前 ---
MANUAL_DATA_FILE = "parameter_map_1_x0=0.02_low parameters epsilon.csv"
# --------------------------------------------------

# 最終的に出力するパラメータマップの画像ファイル名
OUTPUT_IMAGE = "parameter_map_1_x0=0.02_low parameters epsilon.png"

# --- 新しいカテゴリ定義 (8種類) ---
category_labels = {
    0: 'No Sound',
    1: 'Harmonic',
    2: 'Subharmonic',
    3: 'Noisy',
    4: 'Sub -> Harmonic',   # サブ→倍音
    5: 'Noisy -> Sub',      # ノイジー→サブ
    6: 'Noisy -> Harmonic', # ノイジー→倍音
    7: 'Noisy -> No Sound',    # 【追加】ノイジー→無音
    8: 'Harmonic -> No Sound', # 【追加】倍音→無音
    9: 'Freq. Change'       # 周波数時間変化
}

# --- 新しいカラーマップ (8色) ---
# (色はお好みで調整してください)
cmap_colors = [
    '#ffffff', # 0: No Sound (白)
    "#09C3FC", # 1: Harmonic (水色)
    '#FF7F50', # 2: Subharmonic (オレンジ)
    '#DC143C', # 3: Noisy (赤) 
    '#EE82EE', # 4: Sub -> Harmonic (紫)
    "#ce8b6c", # 5: Noisy -> Sub (薄いオレンジ - Subの派生色)
    "#73CBD6", # 6: Noisy -> Harmonic (薄い水色 - Harmonicの派生色)
    "#f3b4b4", # 7: Noisy -> No Sound (非常に薄い赤)
    "#C2ECF1", # 8: Harmonic -> No Sound (非常に薄い水色)
    "#0EDD0E", # 9: Freq. Change (緑)
]
# ------------------------------------


# --- メイン処理 ---
print(f"'{MANUAL_DATA_FILE}' から手動分類データを読み込みます...")

# 1. 手動CSVファイルを読み込む
try:
    df = pd.read_csv(MANUAL_DATA_FILE)
except FileNotFoundError:
    print(f"エラー: {MANUAL_DATA_FILE} が見つかりません。")
    print("ステップ1: 手動で分類したCSVファイルを作成してください。")
    # 動作確認用に、サンプルのCSVファイルを作成します
    dummy_df = pd.DataFrame({
        'epsilon': [1.0e+08, 1.1e+08, 1.0e+08, 1.1e+08],
        'ps': [1.0e+06, 1.0e+06, 1.1e+06, 1.1e+06],
        'category': [6, 3, 2, 4] # サンプルデータ
    })
    dummy_df.to_csv(MANUAL_DATA_FILE, index=False)
    print(f"動作確認用のサンプル {MANUAL_DATA_FILE} を作成しました。")
    exit()

if 'epsilon' not in df.columns or 'ps' not in df.columns or 'category' not in df.columns:
    print("エラー: CSVファイルには 'epsilon', 'ps', 'category' の3列が必要です。")
    exit()

print(f"{len(df)} 件の手動分類データを読み込みました。")

# 2. マトリックスの軸（Epsilon と Ps）を特定
epsilon_axis = sorted(df['epsilon'].unique())
ps_axis = sorted(df['ps'].unique())

if len(epsilon_axis) == 0 or len(ps_axis) == 0:
    print("エラー: CSVに有効なデータがありません。")
    exit()

# 3. マトリックスの作成
result_matrix = np.full((len(ps_axis), len(epsilon_axis)), 0) # デフォルトは 0 (No Sound)

# DataFrameを辞書に変換して高速化
results_map = {}
for index, row in df.iterrows():
    results_map[(row['epsilon'], row['ps'])] = row['category']

# マトリックスを埋める
for i, ps in enumerate(ps_axis):
    for j, eps in enumerate(epsilon_axis):
        # 辞書から (eps, ps) のペアに対応するカテゴリを取得
        # もし手動CSVにデータが欠損していても、デフォルトの 0 が入る
        result_matrix[i, j] = results_map.get((eps, ps), 0)

print("マトリックスの作成が完了しました。")

# 4. カラーマップの描画 (8カテゴリ版)
cmap = mcolors.ListedColormap(cmap_colors)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(13, 10))
plt.imshow(result_matrix, aspect='auto', origin='lower', cmap=cmap, norm=norm,
           extent=[epsilon_axis[0], epsilon_axis[-1], ps_axis[0], ps_axis[-1]])

LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18

plt.xlabel('Epsilon (ε)', fontsize=LABEL_FONTSIZE)
plt.ylabel('Pressure (ps)', fontsize=LABEL_FONTSIZE)

# ここの [::3] の数字を [::2] や [::5] に変えて調整できます
x_tick_locations = epsilon_axis[::3]
x_tick_labels = [f"{eps:.1e}" for eps in x_tick_locations]
plt.xticks(x_tick_locations, x_tick_labels, rotation=45, fontsize=TICK_FONTSIZE)

y_tick_locations = ps_axis[::3]
y_tick_labels = [f"{ps:.1e}" for ps in y_tick_locations]
plt.yticks(y_tick_locations, y_tick_labels, fontsize=TICK_FONTSIZE)

# カラーバーを8カテゴリ用に設定
cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
cbar.set_ticklabels([category_labels[i] for i in range(10)])
#cbar.set_label('Vibration Type', fontsize=LABEL_FONTSIZE)
cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

plt.title('Parameter Map of Birdsong Simulation 1 x0=0.02 low epsilon', fontsize=21)
plt.tight_layout()

plt.savefig(OUTPUT_IMAGE)
print(f"\n手動分類によるパラメータマップを {OUTPUT_IMAGE} という名前で保存しました。")