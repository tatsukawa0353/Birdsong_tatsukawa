# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# --- 【重要】あなたが手動で作成したCSVファイルの名前 ---
MANUAL_DATA_FILE = "parameter_map_1_x0=0.02 add.csv"
# --------------------------------------------------

# 最終的に出力するパラメータマップの画像ファイル名
OUTPUT_IMAGE = "parameter_map_1_x0=0.02 add.png"

# --- 新しいカテゴリ定義 (8種類) ---
category_labels = {
    0: 'No Sound',
    1: 'Harmonic',
    2: 'Subharmonic',
    3: 'Noisy',
    4: 'Others' #その他
    #4: 'Sub -> Harmonic',   # サブ→倍音  　　　　　　　　サブハーモニックへ
    #5: 'Noisy -> Sub',      # ノイジー→サブ　　　　　　　その他
    #6: 'Noisy -> Harmonic', # ノイジー→倍音　　　　　　　その他
    #7: 'Noisy -> No Sound',    # 【追加】ノイジー→無音　ノイジー
    #8: 'Harmonic -> No Sound', # 【追加】倍音→無音　　　倍音
    #9: 'Freq. Change'       # 周波数時間変化　　　　　　その他
}

# --- 新しいカラーマップ (8色) ---
# (色はお好みで調整してください)
cmap_colors = [
    '#ffffff', # 0: No Sound (白)
    "#09C3FC", # 1: Harmonic (水色)
    '#FF7F50', # 2: Subharmonic (オレンジ)
    '#DC143C', # 3: Noisy (赤) 
    "#0EDD0E" # 9: Freq. Change (緑)
    #'#EE82EE', # 4: Sub -> Harmonic (紫)
    #"#ce8b6c", # 5: Noisy -> Sub (薄いオレンジ - Subの派生色)
    #"#73CBD6", # 6: Noisy -> Harmonic (薄い水色 - Harmonicの派生色)
    #"#f3b4b4", # 7: Noisy -> No Sound (非常に薄い赤)
    #"#C2ECF1", # 8: Harmonic -> No Sound (非常に薄い水色)
    #"#0EDD0E", # 9: Freq. Change (緑)
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

# 4. カラーマップの描画 (5カテゴリ版)
cmap = mcolors.ListedColormap(cmap_colors)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5] # 5カテゴリ (6境界)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# --- 【修正点】pcolormesh を使うため、fig と ax を明示的に作成 ---
fig, ax = plt.subplots(figsize=(13, 10))

# --- 【修正点】imshow の代わりに pcolormesh を使用 ---
# pcolormesh は、X軸、Y軸の座標データ（epsilon_axis, ps_axis）と
# Z軸の色データ（result_matrix）を渡して描画します。
# shading='nearest' は、X,Yの座標をピクセルの「中心」として扱います。
im = ax.pcolormesh(
    epsilon_axis, 
    ps_axis, 
    result_matrix, 
    cmap=cmap, 
    norm=norm, 
    shading='nearest'
)

# --- 【修正点】X軸とY軸を「対数スケール」に設定 ---
ax.set_xscale('log')
ax.set_yscale('log')
# ---------------------------------------------

LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18

ax.set_xlabel('Epsilon (ε)', fontsize=LABEL_FONTSIZE)
ax.set_ylabel('Pressure (ps)', fontsize=LABEL_FONTSIZE)


# --- 【修正点】xticks / yticks は対数スケールが自動生成する ---
# (フォントサイズと回転のみ指定)
ax.tick_params(axis='x', labelsize=TICK_FONTSIZE, rotation=45)
ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)

# (もし手動で目盛りを間引きたい場合は、pcolormesh では複雑になるため、
#  まずは自動生成された目盛りで確認してみてください)


# カラーバーを5カテゴリ用に設定
cbar = fig.colorbar(im, ticks=[0, 1, 2, 3, 4]) # plt.colorbar から fig.colorbar に変更
cbar.set_ticklabels([category_labels[i] for i in range(5)])
#cbar.set_label('Vibration Type', fontsize=LABEL_FONTSIZE)
cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

ax.set_title('Parameter Map of Birdsong Simulation : One bronchus x0=0.02 ', fontsize=21)
fig.tight_layout() # plt.tight_layout() から fig.tight_layout() に変更

fig.savefig(OUTPUT_IMAGE) # plt.savefig() から fig.savefig() に変更
print(f"\n手動分類によるパラメータマップを {OUTPUT_IMAGE} という名前で保存しました。")