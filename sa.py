import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# データの読み込み
df1 = pd.read_csv('complexity_data_1.csv')
df2 = pd.read_csv('complexity_data_2.csv')

# データの結合
merged_df = pd.merge(df1, df2, on=['epsilon', 'ps'], suffixes=('_1', '_2'))

# 重複データの削除 (エラー回避)
merged_df = merged_df.drop_duplicates(subset=['epsilon', 'ps'], keep='last')

# 差分の計算
merged_df['diff_raw_complexity'] = merged_df['raw_complexity_1'] - merged_df['raw_complexity_2']

# 個数のカウントと表示
diffs = merged_df['diff_raw_complexity'].dropna()
pos_count = (diffs > 0).sum()
neg_count = (diffs < 0).sum()
zero_count = (diffs == 0).sum()

print(f"Positive counts (Data1 > Data2): {pos_count}")
print(f"Negative counts (Data2 > Data1): {neg_count}")
print(f"Zero counts: {zero_count}")

# ピボットテーブル作成
heatmap_data = merged_df.pivot(index='ps', columns='epsilon', values='diff_raw_complexity')

# 軸の並び替え
heatmap_data.sort_index(ascending=False, inplace=True) # Y軸: 降順 (大きい値が上)
heatmap_data.sort_index(axis=1, ascending=True, inplace=True) # X軸: 昇順

# 描画用データ
data_values = heatmap_data.values
y_labels = heatmap_data.index
x_labels = heatmap_data.columns

# プロット作成 (図のサイズを正方形に設定)
fig, ax = plt.subplots(figsize=(10, 10))

# カラーマップの設定 (0を白にするために対称な範囲を設定)
max_abs = np.nanmax(np.abs(data_values))

# imshowで描画 (aspect='auto' で正方形の枠いっぱいに引き伸ばす)
im = ax.imshow(data_values, cmap='coolwarm', interpolation='nearest', 
               vmin=-max_abs, vmax=max_abs, aspect='auto')

# カラーバー
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 軸目盛りとラベル
ax.set_xticks(np.arange(len(x_labels)))
ax.set_yticks(np.arange(len(y_labels)))
ax.set_xticklabels([f"{x:.1e}" for x in x_labels])
ax.set_yticklabels([f"{y:.1e}" for y in y_labels])

# ラベルの回転
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# 数値の書き込み
for i in range(len(y_labels)):
    for j in range(len(x_labels)):
        val = data_values[i, j]
        if not np.isnan(val):
            # 背景色に応じて文字色を変える
            text_color = "white" if abs(val) > max_abs/2 else "black"
            ax.text(j, i, f"{val:.2e}",
                    ha="center", va="center", color=text_color, fontsize=8)

ax.set_title("2D Heatmap of Raw Complexity Difference (Data 1 - Data 2)")
ax.set_xlabel("Epsilon")
ax.set_ylabel("PS")

plt.tight_layout()
plt.savefig('complexity_difference_heatmap_square.png')
print("Square heatmap saved.")
plt.show()