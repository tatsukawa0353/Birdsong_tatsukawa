from pathlib import Path

import kashika_subharmonic_linear_epsilon as base


base.RESULT_DIRECTORY = Path("single_source_linear_epsilon_results")
base.OUTPUT_DIRECTORY = base.RESULT_DIRECTORY / "figures"
base.CONDITION_FILE = (
    base.RESULT_DIRECTORY / "simulation_single_source_linear_epsilon_conditions.csv"
)


if __name__ == "__main__":
    base.main()
