# 仮想環境に入るため，実行するときにまず source ../venv/bin/activate を
# ターミナルで実行する．

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from scipy.signal import spectrogram


# 既存の描画スタイル
plt.style.use(["science", "ieee"])

# 既存のスペクトログラム設定（変更しない）
NPERSEG = 245760
NOVERLAP = 184320
WINDOW_TYPE = "blackmanharris"
CMAP = "gray_r"
VMIN = -38.002
VMAX = -38
MAX_FREQUENCY = 10000

# 発表・確認用に軸ラベルと目盛りを大きくする
LABEL_SIZE = 70
TICK_SIZE = 70

# シミュレーション時間が従来より長いため横長にする
FIGURE_SIZE = (24, 12)
PAD_INCHES = 0.1
DEFAULT_EPSILON_YLIM = (1.0, 6.5)
PARAMETER_YTICK_STEP = 0.5

RESULT_DIRECTORY = Path("subharmonic_linear_epsilon_results")
OUTPUT_DIRECTORY = RESULT_DIRECTORY / "figures"
CONDITION_FILE = RESULT_DIRECTORY / "simulation_subharmonic_linear_epsilon_conditions.csv"


def apply_axis_style(ax):
    """既存コードと同じ軸・目盛りスタイルを適用する。"""
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


def save_spectrogram(time, pi, output_path):
    """既存設定を維持した横長スペクトログラムを保存する。"""
    total_time = time[-1]
    sampling_rate = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1.0

    # 参考画像と同じく末尾のみゼロパディングする。反射パディングは
    # 使用せず、窓が実データ内で成立する中央の時刻範囲だけを表示する。
    padding_length = NPERSEG
    pi_padded = np.pad(pi, (0, padding_length), mode="constant")

    frequencies, spectrum_time, power = spectrogram(
        pi_padded,
        fs=sampling_rate,
        window=WINDOW_TYPE,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
    )

    valid_time_indices = np.where(spectrum_time <= total_time)[0]
    # 参考画像の処理と同様に、末尾の境界効果が大きい2フレームを除外する。
    if len(valid_time_indices) > 2:
        valid_time_indices = valid_time_indices[:-2]

    # 表示範囲は従来どおり0～10 kHz。描画時のメモリも抑える。
    valid_frequency_indices = frequencies <= MAX_FREQUENCY
    frequencies = frequencies[valid_frequency_indices]
    spectrum_time = spectrum_time[valid_time_indices]
    power = power[valid_frequency_indices][:, valid_time_indices]

    maximum_power = np.max(power)
    normalized_power = power / maximum_power if maximum_power > 0 else power
    db_power = 10 * np.log10(normalized_power + 1e-10)

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
    ax.set_ylabel("Frequency [Hz]", fontsize=LABEL_SIZE)
    ax.set_xlabel("Time [s]", fontsize=LABEL_SIZE)
    ax.set_xlim(spectrum_time[0], spectrum_time[-1])
    time_ticks = np.arange(0.05, 0.40, 0.05)
    ax.set_xticks(time_ticks)
    ax.set_xticklabels([f"{tick:.2f}" for tick in time_ticks])
    ax.set_ylim(0, MAX_FREQUENCY)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=PAD_INCHES)
    plt.close(fig)


def centered_epsilon_axis_limits(epsilon_start, values, minimum_half_span=0.5):
    """ε初期値が縦軸中央に来るように軸範囲を決める。"""
    center = epsilon_start / 1.0e7
    scaled_values = np.asarray(values) / 1.0e7
    maximum_distance = np.max(np.abs(scaled_values - center))
    half_span = max(
        minimum_half_span,
        np.ceil(maximum_distance / 0.5) * 0.5,
    )
    return center - half_span, center + half_span


def epsilon_direction(total_change):
    if total_change > 0.0:
        return "increase"
    if total_change < 0.0:
        return "decrease"
    return "constant"


def build_epsilon_ylim_map(condition_data):
    """同じ初期値・同じ増減方向のε変化図で縦軸をそろえる。"""
    grouped_values = {}
    values_by_start = {}
    for _, row in condition_data.iterrows():
        key = (
            row["epsilon_start"],
            epsilon_direction(row["epsilon_total_change"]),
        )
        grouped_values.setdefault(key, [])
        grouped_values[key].extend([row["epsilon_start"], row["epsilon_end"]])
        values_by_start.setdefault(row["epsilon_start"], [])
        values_by_start[row["epsilon_start"]].extend(
            [row["epsilon_start"], row["epsilon_end"]]
        )

    group_limits = {
        key: centered_epsilon_axis_limits(key[0], np.asarray(values))
        for key, values in grouped_values.items()
    }
    for epsilon_start, values in values_by_start.items():
        constant_key = (epsilon_start, "constant")
        if constant_key in group_limits:
            group_limits[constant_key] = centered_epsilon_axis_limits(
                epsilon_start,
                np.asarray(values),
            )

    ylim_by_filename = {}
    for _, row in condition_data.iterrows():
        key = (
            row["epsilon_start"],
            epsilon_direction(row["epsilon_total_change"]),
        )
        ylim_by_filename[Path(row["output_filename"])] = group_limits[key]

    return ylim_by_filename


def save_epsilon_time_plot(time, epsilon, output_path, epsilon_ylim=DEFAULT_EPSILON_YLIM):
    """CSVに保存されたεと時間の関係を横長の図として保存する。"""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(time, epsilon / 1.0e7, color="black", linewidth=2.0)

    apply_axis_style(ax)
    ax.set_ylabel(r"$\epsilon$ [$\times 10^{7}$ s$^{-2}$]", fontsize=LABEL_SIZE)
    ax.set_xlabel("Time [s]", fontsize=LABEL_SIZE)
    ax.set_xlim(0, time[-1])
    time_ticks = np.arange(0.05, 0.401, 0.05)
    ax.set_xticks(time_ticks)
    ax.set_xticklabels([f"{tick:.2f}" for tick in time_ticks])

    ax.set_ylim(*epsilon_ylim)
    tick_lower = np.ceil(epsilon_ylim[0] / PARAMETER_YTICK_STEP) * PARAMETER_YTICK_STEP
    tick_upper = np.floor(epsilon_ylim[1] / PARAMETER_YTICK_STEP) * PARAMETER_YTICK_STEP
    ax.set_yticks(
        np.arange(tick_lower, tick_upper + 0.001, PARAMETER_YTICK_STEP)
    )
    ax.set_yticklabels([f"{tick:.1f}" for tick in ax.get_yticks()])

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=PAD_INCHES)
    plt.close(fig)


def process_csv(csv_path, epsilon_ylim=DEFAULT_EPSILON_YLIM):
    if not csv_path.exists():
        print(f"スキップ（ファイルなし）: {csv_path}")
        return

    print(f"読み込み中: {csv_path}")
    data = pd.read_csv(
        csv_path,
        usecols=["time", "pi", "epsilon_left"],
        dtype={
            "time": np.float64,
            "pi": np.float64,
            "epsilon_left": np.float64,
        },
    )

    time = data["time"].to_numpy()
    pi = data["pi"].to_numpy()
    epsilon = data["epsilon_left"].to_numpy()

    stem = csv_path.stem
    spectrogram_path = OUTPUT_DIRECTORY / f"{stem}_spectrogram.png"
    epsilon_path = OUTPUT_DIRECTORY / f"{stem}_epsilon_time.png"

    print(f"スペクトログラム作成中: {spectrogram_path}")
    save_spectrogram(time, pi, spectrogram_path)

    print(f"ε－時間図作成中: {epsilon_path}")
    save_epsilon_time_plot(time, epsilon, epsilon_path, epsilon_ylim)

    del data, time, pi, epsilon
    print(f"完了: {csv_path}")


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if not CONDITION_FILE.exists():
        raise FileNotFoundError(
            f"{CONDITION_FILE} が見つかりません。先にシミュレーションを実行してください。"
        )

    condition_data = pd.read_csv(CONDITION_FILE)
    epsilon_ylim_by_filename = build_epsilon_ylim_map(condition_data)
    csv_files = [Path(filename) for filename in condition_data["output_filename"]]

    for csv_path in csv_files:
        process_csv(csv_path, epsilon_ylim_by_filename.get(csv_path, DEFAULT_EPSILON_YLIM))

    print(f"すべての図を {OUTPUT_DIRECTORY} に保存しました。")


if __name__ == "__main__":
    main()
