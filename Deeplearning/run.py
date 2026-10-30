import argparse
import json
from pathlib import Path

from .experiments import run_phase0, run_phase1


def main():
    parser = argparse.ArgumentParser(description="Isolated birdsong experiments")
    parser.add_argument("phase", choices=["phase0", "phase1"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = args.config or root/"configs"/f"{args.phase}.json"
    output = args.output or root/"results"/args.phase
    config = json.loads(config_path.read_text())
    {"phase0": run_phase0, "phase1": run_phase1}[args.phase](config, output)


if __name__ == "__main__":
    main()
