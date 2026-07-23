# 実行例:
#   source ../venv/bin/activate
#   python kashika_distance1.py

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (science スタイルの登録に必要)
from scipy.signal import spectrogram


plt.style.use(["science", "ieee"])

# 既存の時間変化プログラムと同じスペクトログラム設定
NPERSEG = 245760
NOVERLAP = 184320
WINDOW_TYPE = "blackmanharris"
CMAP = "gray_r"
VMIN = -38.002
VMAX = -38
MAX_FREQUENCY = 10000

LABEL_SIZE = 46
TICK_SIZE = 46
# 実音の参考スペクトログラムに近い、ほぼ正方形の縦横比
FIGURE_SIZE = (14, 13)
PAD_INCHES = 0.1

CSV_FILE = Path("simulation_distance1.csv")
OUTPUT_DIRECTORY = Path("distance1_figures")


def apply_axis_style(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2.0)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_SIZE,
        colors="black",
        width=2.0,
        length=10,
        direction="in",
        top=False,
        right=False,
    )


def time_ticks():
    return np.arange(0.0, 0.1201, 0.02)


def save_spectrogram(time, pi, output_path):
    total_time = time[-1]
    sampling_rate = 1.0 / (time[1] - time[0])

    # 既存スクリプトと同じく末尾だけをゼロパディングする。
    pi_padded = np.pad(pi, (0, NPERSEG), mode="constant")
    frequencies, spectrum_time, power = spectrogram(
        pi_padded,
        fs=sampling_rate,
        window=WINDOW_TYPE,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
    )

    valid_time_indices = np.where(spectrum_time <= total_time)[0]
    if len(valid_time_indices) > 2:
        valid_time_indices = valid_time_indices[:-2]

    valid_frequency_indices = frequencies <= MAX_FREQUENCY
    frequencies = frequencies[valid_frequency_indices]
    spectrum_time = spectrum_time[valid_time_indices]
    power = power[valid_frequency_indices][:, valid_time_indices]

    maximum_power = np.max(power)
    normalized_power = power / maximum_power if maximum_power > 0 else power
    db_power = 10.0 * np.log10(normalized_power + 1.0e-10)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.pcolormesh(
        spectrum_time,
        frequencies,
        db_power,
        shading="gouraud",
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
    )
    apply_axis_style(ax)
    ax.set_xlabel("Time [s]", fontsize=LABEL_SIZE)
    ax.set_ylabel("Frequency [Hz]", fontsize=LABEL_SIZE)
    # STFTの解析窓が実際に成立している時刻だけを横軸に表示する。
    # 今回は約 0.012--0.104 s となり、実音の参考図と同じ見せ方になる。
    ax.set_xlim(spectrum_time[0], spectrum_time[-1])
    spectrum_ticks = np.arange(0.02, spectrum_time[-1] + 0.001, 0.02)
    ax.set_xticks(spectrum_ticks)
    ax.set_xticklabels([f"{tick:.2f}" for tick in spectrum_ticks])
    ax.set_ylim(0, MAX_FREQUENCY)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=PAD_INCHES)
    plt.close(fig)
    return spectrum_time[0], spectrum_time[-1]


def save_parameter_time_plot(
    time,
    epsilon_right,
    ps,
    analysis_start,
    analysis_end,
    output_path,
):
    """発声側（右側）の epsilon(t) と共通気嚢圧 ps(t) を描画する。"""
    fig, axes = plt.subplots(
        2,
        1,
        figsize=FIGURE_SIZE,
        sharex=True,
        gridspec_kw={"hspace": 0.15},
    )

    axes[0].plot(time, epsilon_right / 1.0e7, color="black", linewidth=2.0)
    axes[1].plot(time, ps / 1.0e6, color="black", linewidth=2.0)

    # スペクトログラムと同じ解析範囲・目盛りを用いる。
    ticks = np.arange(0.02, analysis_end + 0.001, 0.02)
    for ax in axes:
        apply_axis_style(ax)
        ax.set_xlim(analysis_start, analysis_end)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:.2f}" for tick in ticks])

    axes[0].set_ylabel(r"$\epsilon_{\mathrm{right}}$ [$\times 10^{7}$ s$^{-2}$]", fontsize=LABEL_SIZE)
    axes[1].set_ylabel(r"$p_s$ [$\times 10^{6}$]", fontsize=LABEL_SIZE)
    axes[1].set_xlabel("Time [s]", fontsize=LABEL_SIZE)

    axes[0].set_ylim(3.0, 9.0)
    axes[0].set_yticks(np.arange(3.0, 8.1, 1.0))
    axes[1].set_ylim(2.0, 3.5)
    axes[1].set_yticks(np.arange(2.0, 3.51, 0.5))
    axes[0].set_yticklabels([f"{tick:.1f}" for tick in axes[0].get_yticks()])
    axes[1].set_yticklabels([f"{tick:.1f}" for tick in axes[1].get_yticks()])

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=PAD_INCHES)
    plt.close(fig)


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"{CSV_FILE} が見つかりません。先に birdsong_distance1_sim を実行してください。"
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(
        CSV_FILE,
        usecols=["time", "pi", "epsilon_right", "ps_right"],
        dtype=np.float64,
    )

    time = data["time"].to_numpy()
    pi = data["pi"].to_numpy()
    epsilon_right = data["epsilon_right"].to_numpy()
    ps = data["ps_right"].to_numpy()

    stem = CSV_FILE.stem
    spectrogram_path = OUTPUT_DIRECTORY / f"{stem}_spectrogram.png"
    parameter_path = OUTPUT_DIRECTORY / f"{stem}_epsilon_ps_time.png"

    print(f"スペクトログラム作成中: {spectrogram_path}")
    analysis_start, analysis_end = save_spectrogram(time, pi, spectrogram_path)
    print(f"epsilon・ps－時間図作成中: {parameter_path}")
    save_parameter_time_plot(
        time,
        epsilon_right,
        ps,
        analysis_start,
        analysis_end,
        parameter_path,
    )
    print(f"2つの図を {OUTPUT_DIRECTORY} に保存しました。")


if __name__ == "__main__":
    main()
