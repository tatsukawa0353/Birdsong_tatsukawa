"""Reproduce one legacy condition and benchmark the unchanged integration dt."""
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from scipy.io import wavfile

from .acoustics import distance
from .experiments import write_json
from .model import BirdModel, ROOT, build
from .model_env import AudioDecimator
from .plotting import comparison, single_spectrogram


def legacy_reference():
    """Read one unmodified legacy run without overwriting stored metrics."""
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        exe, csv = directory/"reference", directory/"reference.csv"
        subprocess.run(["g++", "-O3", "-std=c++11", "-I", str(ROOT.parent),
                        str(ROOT/"native/reference.cpp"), str(ROOT.parent/"BirdsongModel_10*10.cpp"),
                        "-o", str(exe)], check=True)
        start = time.perf_counter()
        subprocess.run([str(exe), str(csv), "1200000"], check=True)
        csv_seconds = time.perf_counter()-start
        reference = np.loadtxt(csv, delimiter=",", skiprows=1)[:, 1]
    return reference, csv_seconds


def main():
    build()
    output = ROOT/"results"/"reference"
    output.mkdir(parents=True, exist_ok=True)
    reference, csv_seconds = legacy_reference()
    with BirdModel("A") as model:
        generated = model.advance(.12)
    ref_audio = AudioDecimator().process(reference[1:])
    new_audio = AudioDecimator().process(generated)
    np.savez_compressed(output/"reference.npz", target=ref_audio, audio=new_audio, sample_rate=50000)
    # Float WAV contains model pressure divided by one shared gain, for listening only.
    wavfile.write(output/"reference_synthetic.wav", 50000,
                  (new_audio/max(abs(new_audio))).astype(np.float32))
    comparison(ref_audio, new_audio, 50000, output/"spectrogram.png")
    comparison(ref_audio, new_audio, 50000, output/"legacy_display_scale.png")
    a, b = np.fft.rfft(reference[1:]), np.fft.rfft(generated)
    metrics = dict(csv_seconds=csv_seconds, samples=len(generated), dt=1e-7,
                   fft_relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                   waveform_relative_l2=float(np.linalg.norm(reference[1:]-generated)/np.linalg.norm(generated)),
                   acoustic_error=distance(ref_audio, new_audio, 50000),
                   compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0])
    write_json(output/"regression.json", metrics)
    # Original 10 MHz data with the current display scale and complete windows.
    single_spectrogram(reference, 1e7, output/"legacy_spectrogram.png")
    benchmark = []
    for condition in "ABC":
        for repeat in range(3):
            with BirdModel(condition) as model:
                start = time.perf_counter()
                x = model.advance(.35)
                elapsed = time.perf_counter()-start
                benchmark.append(dict(condition=condition, repeat=repeat, integration_steps=len(x),
                                      seconds=elapsed, pressure_std=float(np.std(x)),
                                      finite=bool(np.isfinite(x).all()), final_state=model.state.tolist()))
    write_json(output/"benchmark.json", benchmark)
    print(metrics)


if __name__ == "__main__":
    main()
