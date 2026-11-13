from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly


CSV_FILE = Path("simulation_distance1.csv")
OUTPUT_WAV = Path("simulation_distance1.wav")
OUTPUT_SAMPLE_RATE = 48_000
PEAK_LEVEL = 0.95
FADE_SECONDS = 0.002


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"{CSV_FILE} が見つかりません。先にシミュレーションを実行してください。"
        )

    data = pd.read_csv(
        CSV_FILE,
        usecols=["time", "pi"],
        dtype={"time": np.float64, "pi": np.float64},
    )
    time = data["time"].to_numpy()
    signal = data["pi"].to_numpy()
    if len(time) < 2:
        raise ValueError("音声化には2点以上のシミュレーションデータが必要です。")

    input_sample_rate = int(round(1.0 / np.median(np.diff(time))))
    common_divisor = gcd(input_sample_rate, OUTPUT_SAMPLE_RATE)
    up = OUTPUT_SAMPLE_RATE // common_divisor
    down = input_sample_rate // common_divisor

    # 定常的な圧力成分は音にならないため除去し、polyphase FIRで
    # アンチエイリアス処理を行いながら可聴用レートへ変換する。
    signal = signal - np.mean(signal)
    audio = resample_poly(signal, up, down)

    # WAVの開始・終了クリックを抑える短い線形フェード。
    fade_samples = min(
        int(round(FADE_SECONDS * OUTPUT_SAMPLE_RATE)),
        len(audio) // 2,
    )
    if fade_samples > 0:
        fade = np.linspace(0.0, 1.0, fade_samples, endpoint=True)
        audio[:fade_samples] *= fade
        audio[-fade_samples:] *= fade[::-1]

    # フェード後の信号を基準に音量を揃え、クリッピングを避ける。
    peak = np.max(np.abs(audio))
    if not np.isfinite(peak) or peak == 0.0:
        raise ValueError("有効な音響信号を生成できませんでした。")
    audio = PEAK_LEVEL * audio / peak

    audio_int16 = np.round(audio * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(OUTPUT_WAV, OUTPUT_SAMPLE_RATE, audio_int16)

    duration = len(audio_int16) / OUTPUT_SAMPLE_RATE
    print(f"入力サンプリング周波数: {input_sample_rate} Hz")
    print(f"出力サンプリング周波数: {OUTPUT_SAMPLE_RATE} Hz")
    print(f"音声時間: {duration:.6f} s")
    print(f"保存しました: {OUTPUT_WAV}")


if __name__ == "__main__":
    main()
