"""Redraw existing spectrograms without changing audio, rewards or training logs."""
import argparse
import os

import numpy as np

from .model import ROOT
from .legacy_paths import PHASE0_ROOT, legacy_phase0_root
from .plotting import MAX_HZ, PHASE0_MAX_HZ, comparison, single_spectrogram


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
                max_hz = (PHASE0_MAX_HZ if path.resolve().is_relative_to(PHASE0_ROOT) or path.resolve().is_relative_to(legacy_phase0_root())
                          else MAX_HZ)
                comparison(data["target"], data["audio"], float(data["sample_rate"]), image,
                           max_hz=max_hz)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--phase0-only", action="store_true",
                       help="Redraw only Phase 0 spectrograms with a 0–2500 Hz frequency axis")
    scope.add_argument("--include-private", action="store_true",
                       help="Also redraw permission-restricted local experiment plots")
    args = parser.parse_args()
    if args.phase0_only:
        print(f"Redrew {redraw(legacy_phase0_root())} historical Phase 0 spectrograms (0–2500 Hz)")
        return
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
