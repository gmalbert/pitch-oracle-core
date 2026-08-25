"""AIC leaderboard — compare model variants by parameter count and fit quality.

Run: python scripts/eval/aic_leaderboard.py
"""

import os
from pathlib import Path

import pandas as pd

from pitch_oracle_core.goal_models import goals_frame_from_historical
from pitch_oracle_core.model_variants import aic_leaderboard


def main():
    data_dir = Path(os.getenv("PITCH_ORACLE_DATA_DIR", "data_files"))
    csv_path = data_dir / "combined_historical_data_with_calculations_new.csv"
    if not csv_path.exists():
        print(f"No historical data at {csv_path}")
        return

    df = pd.read_csv(csv_path, sep="\t")
    goals = goals_frame_from_historical(df)

    print("Fitting model variants...")
    leaderboard = aic_leaderboard(goals)
    print(leaderboard.to_string(index=False))

    output = data_dir / "eval"
    output.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(output / "aic_leaderboard.csv", index=False)
    print(f"\nSaved to {output / 'aic_leaderboard.csv'}")


if __name__ == "__main__":
    main()
