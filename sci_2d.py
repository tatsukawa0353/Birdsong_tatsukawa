import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import matplotlib.colors as colors

# スタイル適用（IEEE論文スタイル）
plt.style.use(['science', 'ieee'])

# ==========================================
# ★設定エリア★
# ==========================================
# (入力CSVファイル, 出力画像ファイル名, タイトル) のリスト
TARGETS = [
    ('sci_data_1_f0=1.0e7.csv', 'sci_2d_1.png', 'SCI Map (f0=1.0e7)'),
    # 他のCSVがあれば追加してください
    ('sci_data_2.csv', 'sci_2d_2.png', 'SCI Map 2'),
    ('sci_data_1_f0=0.1e7.csv', 'sci_2d_1_f0=0.1e7.png', 'SCI MAP(f0=0.1e7)'),
]

DPI = 300

# ★SCIの最大値を固定（CSVの値を見て調整してください。純音なら1.0、倍音が豊かだと5〜10などになります）
VMAX_FIXED = 10.0 

LABEL_SIZE = 32  
FIG_SIZE = (14, 10) 

X_SCALE = 1e6 # 10^6スケール
Y_SCALE = 1e6

CUSTOM_X_TICKS = [0, 50.0e6, 100.0e6, 150.0e6, 200.0e6, 250.0e6, 300.0e6]
CUSTOM_Y_TICKS = [1.0e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6, 6.0e6]

# 横軸（整数優先）
def format_x(x, scale):
    val = round(x / scale, 1)
    if val == int(val): return f"{int(val)}"
    return f"{val:.1f}"

# 縦軸（.0を残す）
def format_y(y, scale):
    return f"{y / scale:.1f}"

# ==========================================
# 関数定義
# ==========================================
def save_2d_heatmap_custom_color(csv_file, output_file, title_text):
    try:
        df = pd.read_csv(csv_file)
        print(f"[{csv_file}] の読み込みに成功しました。")
    except FileNotFoundError:
        print(f"エラー: {csv_file} が見つかりません。")
        return

    # SCI列が存在するか確認
    if 'SCI' not in df.columns: 
        print(f"エラー: {csv_file} に 'SCI' 列が存在しません。")
        return

    # ピボットテーブルの作成（x軸: epsilon, y軸: ps, 値: SCI）
    heatmap_data = df.pivot(index='ps', columns='epsilon', values='SCI')
    
    # ★NaN（ジッタで除外されたノイズ領域）の補間について
    # contourf は NaN を描画しません（背景色になります）。
    # ノイズ領域を空白（白）として明示的に見せたい場合は、ここは補間しない方が良いです。
    # もし、どうしても滑らかに繋げたい場合は以下のコメントアウトを外してください。
    # heatmap_data = heatmap_data.interpolate(limit_direction='both')

    y_coords = heatmap_data.index.values
    x_coords = heatmap_data.columns.values
    data_values = heatmap_data.values

    # カラーマップの作成（ACIの時と同じ 白→水色→赤）
    colors_list = ['#ffffff', '#87CEFA', '#DC143C']
    cmap_custom = colors.LinearSegmentedColormap.from_list("custom_red_blue_white", colors_list)
    
    # ★無効な値（NaN）の背景色を設定（例えばグレーにして「ここはカオス領域」と明示することも可能）
    # cmap_custom.set_bad(color='lightgrey') 

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    X, Y = np.meshgrid(x_coords, y_coords)
    
    # levels で色の境界を指定（0から VMAX_FIXED まで）
    levs = np.linspace(0, VMAX_FIXED, 21)
    
    # 等高線の描画
    im = ax.contourf(X, Y, data_values, levels=levs, cmap=cmap_custom, extend='max')

    # --- 軸目盛り設定 ---
    ax.set_xticks(CUSTOM_X_TICKS)
    ax.set_xticklabels([format_x(v, X_SCALE) for v in CUSTOM_X_TICKS], rotation=0, ha="center")

    ax.set_yticks(CUSTOM_Y_TICKS)
    ax.set_yticklabels([format_y(v, Y_SCALE) for v in CUSTOM_Y_TICKS])

    # ラベル設定
    ax.set_xlabel(r"$\epsilon$ [$\times 10^{6}$ s$^{-2}$]", fontsize=LABEL_SIZE, labelpad=15)
    ax.set_ylabel(r"$p_{s}$ [$\times 10^{6}$ cm s$^{-2}$]", fontsize=LABEL_SIZE, labelpad=15)

    # カラーバーの設定
    # SCIのスケールに合わせてticksを自動または手動で設定
    # 0〜10の範囲なら、2刻み (0, 2, 4, 6, 8, 10) など
    cbar_ticks = np.linspace(0, VMAX_FIXED, 6) 
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.02, ticks=cbar_ticks)
    cbar.set_label('SCI', fontsize=36)
    cbar.ax.tick_params(labelsize=32)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(2.0)

    # 枠線の太さ設定
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.0)

    # 目盛りのデザイン
    ax.tick_params(axis='both', which='major', labelsize=32, width=2.0, length=10, 
                   direction='in', top=False, right=False)
    
    cbar.ax.tick_params(labelsize=32, width=2.0, length=8, direction='in')

    plt.savefig(output_file, bbox_inches='tight', dpi=DPI)
    plt.close()
    print(f"画像 {output_file} を保存しました。")

# ==========================================
# 実行
# ==========================================
print("グラフ描画を開始します...")
for csv_file, out_file, title in TARGETS:
    save_2d_heatmap_custom_color(csv_file, out_file, title)
print("完了しました。")