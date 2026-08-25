"""MatchFlow shots pipeline — extract and score shots from event data.

Run: python scripts/events/matchflow_shots.py
"""

import os
from pathlib import Path

import pandas as pd

from pitch_oracle_core.events.matchflow import statsbomb_shots


def main():
    data_dir = Path(os.getenv("PITCH_ORACLE_DATA_DIR", "data_files"))
    statsbomb_dir = data_dir / "statsbomb" / "open-data" / "data"
    if not statsbomb_dir.exists():
        print(f"No StatsBomb data at {statsbomb_dir}")
        return

    print("Streaming shots from StatsBomb open data...")
    shots = statsbomb_shots(statsbomb_dir)
    print(f"  {len(shots)} shots extracted")

    output = data_dir / "events"
    output.mkdir(parents=True, exist_ok=True)
    shots.to_parquet(output / "statsbomb_shots.parquet", index=False)
    print(f"Saved to {output / 'statsbomb_shots.parquet'}")


if __name__ == "__main__":
    main()
