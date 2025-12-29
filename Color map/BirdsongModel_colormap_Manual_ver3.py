# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# --- 【重要】あなたが手動で作成したCSVファイルの名前 ---
MANUAL_DATA_FILE = "parameter_map_x0=0.02_linked_eps.csv"
# --------------------------------------------------

# 最終的に出力するパラメータマップの画像ファイル名
OUTPUT_IMAGE = "parameter_map_x0=0.02_linked_eps*0.5.png"

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

# 4. カラーマップの描画 (横軸：実数値スケール、縦軸：均等スケール)
cmap = mcolors.ListedColormap(cmap_colors)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5] # 5カテゴリ (6境界)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(13, 10))

# --- 【重要】ハイブリッド描画の準備 ---
# 縦軸用のダミー座標 (0, 1, 2...) を作成 -> これで縦は均等幅になる
y_indices = np.arange(len(ps_axis))

# --- 【重要】pcolormesh で描画 ---
# X軸には「実数値(epsilon_axis)」を、Y軸には「インデックス(y_indices)」を渡す
# shading='nearest' にすることで、各点がブロックの中心になるよう自動調整されます
im = ax.pcolormesh(epsilon_axis, y_indices, result_matrix, 
                   cmap=cmap, norm=norm, shading='nearest')

LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18

ax.set_xlabel('Left Epsilon (ε)', fontsize=LABEL_FONTSIZE)
ax.set_ylabel('Pressure (ps)', fontsize=LABEL_FONTSIZE)

# --- X軸の設定 (実数値スケールなので Matplotlib が自動調整するが、ラベルを間引く) ---
# データ間隔が広がる後半はブロックが横に広くなり、前半は狭くなります
# 重なりを防ぐため、自動目盛りではなく手動で設定します

x_ticks = epsilon_axis[::3] # 3つおきに表示
ax.set_xticks(x_ticks)
ax.set_xticklabels([f"{eps:.1e}" for eps in x_ticks], rotation=45, fontsize=TICK_FONTSIZE)

# --- Y軸の設定 (インデックス座標なので、ラベルを手動で貼り付ける) ---
# 座標は 0, 1, 2... なので、目盛りもその位置に合わせる
y_tick_indices = y_indices[::3] # 3つおき
y_tick_labels = [f"{ps_axis[i]:.1e}" for i in y_tick_indices]

ax.set_yticks(y_tick_indices)
ax.set_yticklabels(y_tick_labels, fontsize=TICK_FONTSIZE)


# カラーバーの設定
cbar = fig.colorbar(im, ticks=[0, 1, 2, 3, 4])
cbar.set_ticklabels([category_labels[i] for i in range(5)])
cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

ax.set_title('Parameter Map of Birdsong Simulation : Two bronchi x0=0.02 ', fontsize=21)
fig.tight_layout()

fig.savefig(OUTPUT_IMAGE)
print(f"\n手動分類によるパラメータマップを {OUTPUT_IMAGE} という名前で保存しました。")