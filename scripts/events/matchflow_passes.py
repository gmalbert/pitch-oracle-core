"""MatchFlow passes pipeline — extract passes from event data.

Run: python scripts/events/matchflow_passes.py
"""

import os
from pathlib import Path

import pandas as pd

from pitch_oracle_core.events.matchflow import statsbomb_passes


def main():
    data_dir = Path(os.getenv("PITCH_ORACLE_DATA_DIR", "data_files"))
    statsbomb_dir = data_dir / "statsbomb" / "open-data" / "data"
    if not statsbomb_dir.exists():
        print(f"No StatsBomb data at {statsbomb_dir}")
        return

    print("Streaming passes from StatsBomb open data...")
    passes = statsbomb_passes(statsbomb_dir)
    print(f"  {len(passes)} passes extracted")

    output = data_dir / "events"
    output.mkdir(parents=True, exist_ok=True)
    passes.to_parquet(output / "statsbomb_passes.parquet", index=False)
    print(f"Saved to {output / 'statsbomb_passes.parquet'}")


if __name__ == "__main__":
    main()
