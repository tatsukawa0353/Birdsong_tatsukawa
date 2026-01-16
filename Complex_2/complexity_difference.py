import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# データの読み込み
# (ファイル名はアップロードされたものに合わせています)
df1 = pd.read_csv('complexity_data_linked_eps_0.6.csv')
df2 = pd.read_csv('complexity_data_1_f0=4.9e4.csv')

# データの結合
merged_df = pd.merge(df1, df2, on=['epsilon', 'ps'], suffixes=('_1', '_2'))

# 重複データの削除
merged_df = merged_df.drop_duplicates(subset=['epsilon', 'ps'], keep='last')

# 差分の計算
merged_df['diff_raw_complexity'] = merged_df['raw_complexity_1'] - merged_df['raw_complexity_2']

# --- 個数のカウントと表示 ---
diffs = merged_df['diff_raw_complexity'].dropna()
pos_count = (diffs > 0).sum()
neg_count = (diffs < 0).sum()
zero_count = (diffs == 0).sum()

print(f"Positive counts (Data1 > Data2): {pos_count}")
print(f"Negative counts (Data2 > Data1): {neg_count}")
print(f"Zero counts: {zero_count}")
# ---------------------------

# ピボットテーブルの作成
heatmap_data = merged_df.pivot(index='ps', columns='epsilon', values='diff_raw_complexity')

# 軸の並び替え
heatmap_data.sort_index(ascending=False, inplace=True)
heatmap_data.sort_index(axis=1, ascending=True, inplace=True)

# プロット用データ準備
data_values = heatmap_data.values
y_labels = heatmap_data.index
x_labels = heatmap_data.columns

# プロット
fig, ax = plt.subplots(figsize=(16, 8))

# カラーマップの設定 (0中心)
max_abs = np.nanmax(np.abs(data_values))
im = ax.imshow(data_values, cmap='coolwarm', interpolation='nearest', vmin=-max_abs, vmax=max_abs)

# カラーバー
cbar = ax.figure.colorbar(im, ax=ax)

# 軸の設定
ax.set_xticks(np.arange(len(x_labels)))
ax.set_yticks(np.arange(len(y_labels)))
ax.set_xticklabels([f"{x:.1e}" for x in x_labels])
ax.set_yticklabels([f"{y:.1e}" for y in y_labels])

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# 数値の書き込み
for i in range(len(y_labels)):
    for j in range(len(x_labels)):
        val = data_values[i, j]
        if not np.isnan(val):
            text_color = "white" if abs(val) > max_abs/2 else "black"
            ax.text(j, i, f"{val:.1e}",
                    ha="center", va="center", color=text_color, fontsize=8)

ax.set_title("2D Heatmap of Raw Complexity Difference")
ax.set_xlabel("Epsilon")
ax.set_ylabel("PS")

plt.tight_layout()
plt.savefig('complexity_difference_heatmap_eps*0.6.png')
print("保存完了: complexity_difference_heatmap.png")