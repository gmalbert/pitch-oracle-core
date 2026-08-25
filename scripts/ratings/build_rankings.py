"""Build ratings from historical data — Elo, Massey, Colley, Pi.

Run: python scripts/ratings/build_rankings.py
"""

import os
from pathlib import Path

import pandas as pd

from pitch_oracle_core.goal_models import goals_frame_from_historical
from pitch_oracle_core.ratings import (
    build_combined_rankings,
    compute_elo_ratings,
    compute_pi_ratings,
)


def main():
    data_dir = Path(os.getenv("PITCH_ORACLE_DATA_DIR", "data_files"))
    csv_path = data_dir / "combined_historical_data_with_calculations_new.csv"
    if not csv_path.exists():
        print(f"No historical data at {csv_path}")
        return

    df = pd.read_csv(csv_path, sep="\t")
    goals = goals_frame_from_historical(df)

    print("Computing ratings...")
    rankings = build_combined_rankings(goals)
    print(rankings[["team", "elo", "elo_rank", "massey_rank", "colley_rank", "pi_rank"]].to_string(index=False))

    output = data_dir / "ratings"
    output.mkdir(parents=True, exist_ok=True)
    rankings.to_parquet(output / "all_rankings.parquet", index=False)

    # Elo history
    _, elo_history = compute_elo_ratings(goals)
    pd.DataFrame(elo_history).to_parquet(output / "elo_history.parquet", index=False)

    # Pi history
    _, pi_history = compute_pi_ratings(goals)
    pd.DataFrame(pi_history).to_parquet(output / "pi_history.parquet", index=False)

    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
