"""Fit xT model per (competition, season) from event data.

Run: python scripts/xt/fit_xt.py
"""

import os
from pathlib import Path

import pandas as pd

from pitch_oracle_core.xt import fit_xt_model


def main():
    data_dir = Path(os.getenv("PITCH_ORACLE_DATA_DIR", "data_files"))
    events_path = data_dir / "events" / "statsbomb_shots.parquet"
    if not events_path.exists():
        print(f"No events at {events_path}. Run matchflow_shots.py first.")
        return

    events = pd.read_parquet(events_path)
    print(f"Loaded {len(events)} events")

    output = data_dir / "xt"
    output.mkdir(parents=True, exist_ok=True)

    # Fit on all available data (or per-competition if competition_id present)
    model = fit_xt_model(events, output_path=output / "xt_v1.npz")
    print(f"xT model fitted: {model.n_cols}x{model.n_rows} grid")
    print(f"Saved to {output / 'xt_v1.npz'}")


if __name__ == "__main__":
    main()
