import argparse
import json
from pathlib import Path

from .experiments import run_phase0, run_phase1
from .legacy_paths import legacy_phase0_root, PHASE0_ROOT


def main():
    parser = argparse.ArgumentParser(description="Isolated birdsong experiments")
    parser.add_argument("phase", choices=["phase0", "phase1"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = args.config or root/"configs"/f"{args.phase}.json"
    output = args.output or (legacy_phase0_root() if args.phase=="phase0" else root/"results"/args.phase)
    if args.phase=="phase0" and not output.resolve().is_relative_to(PHASE0_ROOT):
        parser.error("New Phase 0 experiments must remain under phase0_fixed_target_20261108")
    if args.phase=="phase0" and output.exists():
        parser.error("Historical Phase 0 is preserved; specify a new output under the unified Phase 0 parent, or use issue15")
    config = json.loads(config_path.read_text())
    {"phase0": run_phase0, "phase1": run_phase1}[args.phase](config, output)


if __name__ == "__main__":
    main()
