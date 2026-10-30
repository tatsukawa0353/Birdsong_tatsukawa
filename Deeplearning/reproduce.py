"""Reproduce one legacy condition and benchmark the unchanged integration dt."""
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .acoustics import distance
from .experiments import write_json
from .model import BirdModel, ROOT, build
from .model_env import AudioDecimator
from .plotting import comparison


def main():
    build()
    output = ROOT/"results"/"reference"
    output.mkdir(parents=True, exist_ok=True)
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
    with BirdModel("A") as model:
        generated = model.advance(.12)
    ref_audio = AudioDecimator().process(reference[1:])
    new_audio = AudioDecimator().process(generated)
    np.savez_compressed(output/"reference.npz", target=ref_audio, audio=new_audio, sample_rate=50000)
    # Float WAV contains model pressure divided by one shared gain, for listening only.
    wavfile.write(output/"reference_synthetic.wav", 50000,
                  (new_audio/max(abs(new_audio))).astype(np.float32))
    comparison(ref_audio, new_audio, 50000, output/"spectrogram.png")
    comparison(ref_audio, new_audio, 50000, output/"legacy_display_scale.png", legacy_scale=True)
    a, b = np.fft.rfft(reference[1:]), np.fft.rfft(generated)
    metrics = dict(csv_seconds=csv_seconds, samples=len(generated), dt=1e-7,
                   fft_relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                   waveform_relative_l2=float(np.linalg.norm(reference[1:]-generated)/np.linalg.norm(generated)),
                   acoustic_error=distance(ref_audio, new_audio, 50000),
                   compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0])
    write_json(output/"regression.json", metrics)
    # Single-panel legacy spectrogram: same PSD, padding, boundary and display settings.
    from .plotting import plt
    padded = np.pad(reference, (0, 245760))
    f, t, p = signal.spectrogram(padded, fs=1e7, window="blackmanharris",
                                nperseg=245760, noverlap=184320)
    keep = np.where(t <= .12)[0][:-2]
    p = p[:, keep]
    db = 10*np.log10(p/p.max()+1e-10)
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.pcolormesh(t[keep], f, db, shading="gouraud", cmap="gray_r", vmin=-38.002, vmax=-38)
        ax.set(xlim=(0, .12), ylim=(0, 10000), xticks=[.02, .04, .06, .08, .10])
        ax.set_xlabel("Time [s]", fontsize=46)
        ax.set_ylabel("Frequency [Hz]", fontsize=46)
        ax.tick_params(labelsize=46, width=2, length=10, direction="in", top=False, right=False)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        fig.tight_layout()
        fig.savefig(output/"legacy_spectrogram.png", bbox_inches="tight", pad_inches=.1, dpi=120)
        plt.close(fig)
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
