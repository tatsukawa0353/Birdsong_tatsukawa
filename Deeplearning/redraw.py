"""Redraw existing spectrograms without changing audio, rewards or training logs."""
import argparse
import os

import numpy as np

from .model import ROOT
from .plotting import comparison, single_spectrogram


def redraw(directory):
    count = 0
    for path in sorted(directory.rglob("*.npz")):
        if path.name == "rollout.npz":
            images = [path.parent/"spectrogram.png"]
        elif path.name == "reference.npz":
            images = [path.parent/"spectrogram.png", path.parent/"legacy_display_scale.png"]
        else:
            # Preserve the existing artifact set; do not add a plot for every archive.
            images = [path.with_suffix(".png")] if path.with_suffix(".png").is_file() else []
        if not images:
            continue
        with np.load(path, allow_pickle=False) as data:
            for image in images:
                comparison(data["target"], data["audio"], float(data["sample_rate"]), image)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-private", action="store_true",
                        help="Also redraw permission-restricted local experiment plots")
    args = parser.parse_args()
    public_count = redraw(ROOT/"results")
    if (ROOT/"results/reference/reference.npz").is_file():
        from .reproduce import legacy_reference
        reference, _ = legacy_reference()
        single_spectrogram(reference, 1e7, ROOT/"results/reference/legacy_spectrogram.png")
        public_count += 1
    print(f"Redrew {public_count} synthetic spectrograms")
    if args.include_private:
        previous = os.umask(0o077)
        try:
            private_count = redraw(ROOT/"private_runs")
        finally:
            os.umask(previous)
        print(f"Redrew {private_count} private spectrograms locally")


if __name__ == "__main__":
    main()
