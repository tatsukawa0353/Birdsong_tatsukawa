from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from scipy.signal import spectrogram


CSV_FILE = Path("simulation_distance1.csv")
OUTPUT_FILE = Path("distance1_figures/simulation_distance1_psd.png")
NORMALIZED_OUTPUT_FILE = Path(
    "distance1_figures/simulation_distance1_psd_normalized.png"
)

NPERSEG = 245760
NOVERLAP = 184320
WINDOW = "blackmanharris"
MAX_FREQUENCY = 10000.0
FIGURE_SIZE = (18, 10)
LABEL_SIZE = 46
TICK_SIZE = 46


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


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    data = pd.read_csv(
        CSV_FILE,
        usecols=["time", "pi"],
        dtype={"time": np.float64, "pi": np.float64},
    )
    time = data["time"].to_numpy()
    pi = data["pi"].to_numpy()
    if len(time) < 2:
        raise ValueError("PSD計算には2点以上のデータが必要です。")

    sampling_rate = 1.0 / np.median(np.diff(time))
    pi = pi - np.mean(pi)
    pi_padded = np.pad(pi, (0, NPERSEG), mode="constant")

    frequencies, _, power = spectrogram(
        pi_padded,
        fs=sampling_rate,
        window=WINDOW,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
    )
    mean_power = np.mean(power, axis=1)
    if np.max(mean_power) <= 0.0:
        raise ValueError("PSDを計算できる有効な信号がありません。")

    valid = frequencies <= MAX_FREQUENCY
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(
        frequencies[valid],
        mean_power[valid],
        color="black",
        linewidth=2.0,
    )
    apply_axis_style(ax)
    ax.set_xlabel("Frequency [Hz]", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"Power Spectral Density [$\pi^2$/Hz]", fontsize=LABEL_SIZE)
    ax.set_xlim(0.0, MAX_FREQUENCY)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks(np.arange(0.0, MAX_FREQUENCY + 1.0, 2000.0))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"保存しました: {OUTPUT_FILE}")

    relative_power_db = 10.0 * np.log10(
        mean_power / np.max(mean_power) + 1.0e-10
    )
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(
        frequencies[valid],
        relative_power_db[valid],
        color="black",
        linewidth=2.0,
    )
    apply_axis_style(ax)
    ax.set_xlabel("Frequency [Hz]", fontsize=LABEL_SIZE)
    ax.set_ylabel("Relative Power [dB]", fontsize=LABEL_SIZE)
    ax.set_xlim(0.0, MAX_FREQUENCY)
    ax.set_ylim(-100.0, 5.0)
    ax.set_xticks(np.arange(0.0, MAX_FREQUENCY + 1.0, 2000.0))
    ax.set_yticks(np.arange(-100.0, 1.0, 20.0))

    fig.tight_layout()
    fig.savefig(NORMALIZED_OUTPUT_FILE, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"保存しました: {NORMALIZED_OUTPUT_FILE}")


if __name__ == "__main__":
    plt.style.use(["science", "ieee"])
    main()
