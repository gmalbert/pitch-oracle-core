"""Walk-forward evaluation harness — CLI entry point.

Run: python scripts/eval/walk_forward.py
"""

import os
from pathlib import Path

import pandas as pd
from penaltyblog.models import DixonColesGoalModel

from pitch_oracle_core.evaluation.walk_forward import walk_forward_evaluate
from pitch_oracle_core.goal_models import goals_frame_from_historical


def main():
    data_dir = Path(os.getenv("PITCH_ORACLE_DATA_DIR", "data_files"))
    csv_path = data_dir / "combined_historical_data_with_calculations_new.csv"
    if not csv_path.exists():
        print(f"No historical data at {csv_path}")
        return

    df = pd.read_csv(csv_path, sep="\t")
    goals = goals_frame_from_historical(df)

    def factory(train):
        from pitch_oracle_core.goal_models import fit_dixon_coles
        return fit_dixon_coles(train)

    result = walk_forward_evaluate(
        goals, factory, window=1500, horizon=50, date_col="date"
    )
    print(f"Walk-forward: {result.n_folds} folds")
    print(f"  Mean Brier:    {result.mean_brier:.4f}")
    print(f"  Mean Log Loss: {result.mean_log_loss:.4f}")
    print(f"  Mean RPS:      {result.mean_rps:.4f}")

    output = data_dir / "eval"
    output.mkdir(parents=True, exist_ok=True)
    result.to_dataframe().to_parquet(output / "walk_forward_dc.parquet", index=False)
    print(f"Saved to {output / 'walk_forward_dc.parquet'}")


if __name__ == "__main__":
    main()
