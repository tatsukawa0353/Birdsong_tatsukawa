from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from scipy.stats import entropy


CSV_FILE = Path("simulation_distance1.csv")
OUTPUT_FILE = Path("simulation_distance1_aci_se.csv")

NPERSEG = 245760
NOVERLAP = 184320
WINDOW = "blackmanharris"
MIN_FREQUENCY = 250.0
MAX_FREQUENCY = 12000.0
ACI_NOISE_THRESHOLD_RATIO = 0.01


def calculate_indices(csv_path):
    data = pd.read_csv(
        csv_path,
        usecols=["time", "pi"],
        dtype={"time": np.float64, "pi": np.float64},
    )
    time = data["time"].to_numpy()
    pi = data["pi"].to_numpy()
    if len(time) < NPERSEG:
        raise ValueError(f"{NPERSEG}点以上のデータが必要です。")

    sampling_rate = 1.0 / np.median(np.diff(time))
    pi = pi - np.mean(pi)
    frequencies, _, power = spectrogram(
        pi,
        fs=sampling_rate,
        window=WINDOW,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
    )

    band = (
        (frequencies >= MIN_FREQUENCY)
        & (frequencies <= MAX_FREQUENCY)
    )
    band_power = power[band, :]
    if band_power.size == 0 or not np.any(band_power > 0.0):
        return sampling_rate, power.shape[1], 0.0, 0.0

    # ACI: frequency-bin-wise temporal absolute change divided by total power.
    sum_power = np.sum(band_power, axis=1)
    sum_difference = np.sum(np.abs(np.diff(band_power, axis=1)), axis=1)
    threshold = np.max(sum_power) * ACI_NOISE_THRESHOLD_RATIO
    with np.errstate(divide="ignore", invalid="ignore"):
        aci_per_bin = np.divide(
            sum_difference,
            sum_power,
            out=np.zeros_like(sum_difference),
            where=sum_power >= threshold,
        )
    aci = float(np.sum(aci_per_bin))

    # SE: normalized Shannon entropy of the time-averaged spectrum.
    mean_spectrum = np.mean(band_power, axis=1)
    spectrum_sum = np.sum(mean_spectrum)
    if spectrum_sum > 0.0 and len(mean_spectrum) > 1:
        probability = mean_spectrum / spectrum_sum
        spectral_entropy = float(
            entropy(probability, base=2) / np.log2(len(probability))
        )
    else:
        spectral_entropy = 0.0

    return sampling_rate, power.shape[1], aci, spectral_entropy


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    sampling_rate, time_frames, aci, spectral_entropy = calculate_indices(CSV_FILE)
    result = pd.DataFrame(
        [
            {
                "filename": CSV_FILE.name,
                "sampling_rate_hz": sampling_rate,
                "nperseg": NPERSEG,
                "noverlap": NOVERLAP,
                "min_frequency_hz": MIN_FREQUENCY,
                "max_frequency_hz": MAX_FREQUENCY,
                "spectrogram_time_frames": time_frames,
                "aci_noise_threshold_ratio": ACI_NOISE_THRESHOLD_RATIO,
                "ACI": aci,
                "SE": spectral_entropy,
            }
        ]
    )
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"ACI = {aci:.12f}")
    print(f"SE  = {spectral_entropy:.12f}")
    print(f"保存しました: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
