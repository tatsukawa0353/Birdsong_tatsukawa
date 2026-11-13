from pathlib import Path

import calculate_distance1_aci_se as indices
import generate_distance1_audio as audio
import kashika_distance1 as visualization
import plot_distance1_psd as psd


CSV_FILE = Path("simulation_distance1_single_source.csv")
OUTPUT_DIRECTORY = Path("distance1_figures")
STEM = CSV_FILE.stem


def main():
    visualization.CSV_FILE = CSV_FILE
    visualization.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
    visualization.main()

    audio.CSV_FILE = CSV_FILE
    audio.OUTPUT_WAV = Path(f"{STEM}.wav")
    audio.main()

    indices.CSV_FILE = CSV_FILE
    indices.OUTPUT_FILE = Path(f"{STEM}_aci_se.csv")
    indices.main()

    psd.CSV_FILE = CSV_FILE
    psd.OUTPUT_FILE = OUTPUT_DIRECTORY / f"{STEM}_psd.png"
    psd.NORMALIZED_OUTPUT_FILE = (
        OUTPUT_DIRECTORY / f"{STEM}_psd_normalized.png"
    )
    psd.main()


if __name__ == "__main__":
    main()
