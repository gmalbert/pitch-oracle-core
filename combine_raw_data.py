"""Download and combine football-data.co.uk history for any configured league."""

from pathlib import Path
from typing import Iterable
import pandas as pd

from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.leagues import get_league_config


def combine_raw_data(
    league: LeagueConfig | str = "epl",
    seasons: Iterable[str] = ("2122", "2223", "2324", "2425", "2526"),
    output_dir: str | Path = "data_files",
) -> pd.DataFrame:
    config = get_league_config(league) if isinstance(league, str) else league
    frames = []
    for season in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{config.football_data_div}.csv"
        try:
            frame = pd.read_csv(url)
            frame["League"] = config.key
            frames.append(frame)
        except Exception as exc:
            print(f"Error loading {url}: {exc}")
    if not frames:
        raise RuntimeError(f"No historical data was downloaded for {config.key}")
    result = pd.concat(frames, ignore_index=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "combined_historical_data.csv", sep="\t", index=False)
    return result


if __name__ == "__main__":
    combine_raw_data()

