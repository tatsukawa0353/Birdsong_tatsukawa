from pathlib import Path

import kashika_linear_ps as base


base.RESULT_DIRECTORY = Path("single_source_linear_ps_results")
base.OUTPUT_DIRECTORY = base.RESULT_DIRECTORY / "figures"
base.CONDITION_FILE = (
    base.RESULT_DIRECTORY / "simulation_single_source_linear_ps_conditions.csv"
)


if __name__ == "__main__":
    base.main()
