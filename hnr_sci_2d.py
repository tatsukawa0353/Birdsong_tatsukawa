import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "ieee"])
except ImportError:
    plt.style.use("default")


# ==========================================
# 設定エリア
# ==========================================

INPUT_DIR = "HNR_index_比較"

TARGETS = [
    (
        "hnr_sci_aci_se_1_f0=0.05e7.csv",
        "hnr_sci_2d_1_f0=0.05e7.png",
        "HNR-filtered SCI MAP (f0=0.05e7)",
    ),
    (
        "hnr_sci_aci_se_1_f0=0.1e7.csv",
        "hnr_sci_2d_1_f0=0.1e7.png",
        "HNR-filtered SCI MAP (f0=0.1e7)",
    ),
    (
        "hnr_sci_aci_se_1_f0=0.4e7.csv",
        "hnr_sci_2d_1_f0=0.4e7.png",
        "HNR-filtered SCI MAP (f0=0.4e7)",
    ),
    (
        "hnr_sci_aci_se_1_f0=0.7e7.csv",
        "hnr_sci_2d_1_f0=0.7e7.png",
        "HNR-filtered SCI MAP (f0=0.7e7)",
    ),
    (
        "hnr_sci_aci_se_1_f0=1.0e7.csv",
        "hnr_sci_2d_1_f0=1.0e7.png",
        "HNR-filtered SCI MAP (f0=1.0e7)",
    ),
    (
        "hnr_sci_aci_se_1_f0=4.9e4.csv",
        "hnr_sci_2d_1_f0=4.9e4.png",
        "HNR-filtered SCI MAP (f0=4.9e4)",
    ),
    (
        "hnr_sci_aci_se_2.csv",
        "hnr_sci_2d_2.png",
        "HNR-filtered SCI MAP 2",
    ),
]

DPI = 300
VMAX_FIXED = 10.0
HNR_FILTERED_COLOR = "#E6E6E6"

LABEL_SIZE = 32
FIG_SIZE = (14, 10)

X_SCALE = 1e6
Y_SCALE = 1e6

CUSTOM_X_TICKS = [0, 50.0e6, 100.0e6, 150.0e6, 200.0e6, 250.0e6, 300.0e6]
CUSTOM_Y_TICKS = [1.0e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6, 6.0e6]


def format_x(x, scale):
    val = round(x / scale, 1)
    if val == int(val):
        return f"{int(val)}"
    return f"{val:.1f}"


def format_y(y, scale):
    return f"{y / scale:.1f}"


def save_2d_heatmap(csv_file, output_file, title_text):
    if not os.path.exists(csv_file):
        print(f"エラー: {csv_file} が見つかりません。")
        return

    df = pd.read_csv(csv_file)
    print(f"[{csv_file}] の読み込みに成功しました。")

    required_columns = {"epsilon", "ps", "SCI"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"エラー: {csv_file} に必要な列がありません: {sorted(missing_columns)}")
        return

    heatmap_data = df.pivot(index="ps", columns="epsilon", values="SCI")
    y_coords = heatmap_data.index.values
    x_coords = heatmap_data.columns.values
    data_values = heatmap_data.values

    colors_list = ["#ffffff", "#87CEFA", "#DC143C"]
    cmap_custom = colors.LinearSegmentedColormap.from_list(
        "custom_red_blue_white",
        colors_list,
    )
    # HNRフィルタで除外されたNaN領域は、有効値0の白と区別する。
    cmap_custom.set_bad(color=HNR_FILTERED_COLOR, alpha=1.0)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # contourfのマスク領域が透過描画される場合にも同じ色を表示する。
    ax.set_facecolor(HNR_FILTERED_COLOR)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    levels = np.linspace(0, VMAX_FIXED, 29)

    im = ax.contourf(
        x_grid,
        y_grid,
        data_values,
        levels=levels,
        cmap=cmap_custom,
        extend="max",
    )

    ax.set_xticks(CUSTOM_X_TICKS)
    ax.set_xticklabels([format_x(v, X_SCALE) for v in CUSTOM_X_TICKS], rotation=0, ha="center")

    ax.set_yticks(CUSTOM_Y_TICKS)
    ax.set_yticklabels([format_y(v, Y_SCALE) for v in CUSTOM_Y_TICKS])

    ax.set_xlabel(r"$\epsilon$ [$\times 10^{6}$ s$^{-2}$]", fontsize=LABEL_SIZE, labelpad=15)
    ax.set_ylabel(r"$p_{s}$ [$\times 10^{6}$ cm s$^{-2}$]", fontsize=LABEL_SIZE, labelpad=15)
    cbar_ticks = np.arange(0, VMAX_FIXED + 0.1, 2.0)
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.02, ticks=cbar_ticks)
    cbar.set_label("SCI", fontsize=36)
    cbar.ax.tick_params(labelsize=32)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(2.0)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2.0)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=32,
        width=2.0,
        length=10,
        direction="in",
        top=False,
        right=False,
    )
    cbar.ax.tick_params(labelsize=32, width=2.0, length=8, direction="in")

    plt.savefig(output_file, bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"画像 {output_file} を保存しました。")


def main():
    print("HNRフィルタ後SCIの2Dマップ描画を開始します...")
    for csv_name, output_name, title in TARGETS:
        csv_path = os.path.join(INPUT_DIR, csv_name)
        output_path = os.path.join(INPUT_DIR, output_name)
        save_2d_heatmap(csv_path, output_path, title)
    print("完了しました。")


if __name__ == "__main__":
    main()
