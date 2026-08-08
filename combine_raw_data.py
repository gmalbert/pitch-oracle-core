"""Download and combine football-data.co.uk history for any configured league."""

from datetime import date
from pathlib import Path
from typing import Iterable
import os
import pandas as pd

from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.features import parse_match_dates
from pitch_oracle_core.leagues import get_league_config


def recent_season_codes(today: date | None = None, count: int = 5) -> tuple[str, ...]:
    """Return football-data season codes ending with the current league season."""
    today = today or date.today()
    current_start = today.year if today.month >= 7 else today.year - 1
    starts = range(current_start - count + 1, current_start + 1)
    return tuple(f"{year % 100:02d}{(year + 1) % 100:02d}" for year in starts)


def combine_raw_data(
    league: LeagueConfig | str = "epl",
    seasons: Iterable[str] | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    config = get_league_config(league) if isinstance(league, str) else league
    seasons = tuple(seasons or recent_season_codes())
    frames = []
    for season in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{config.football_data_div}.csv"
        try:
            frame = pd.read_csv(url)
            if "Date" not in frame:
                raise ValueError(f"Downloaded season {season} has no Date column")
            parsed_dates = parse_match_dates(frame["Date"])
            if parsed_dates.isna().any():
                raise ValueError(f"Downloaded season {season} contains invalid match dates")
            frame["Date"] = parsed_dates.dt.strftime("%Y-%m-%d")
            frame["League"] = config.key
            frames.append(frame)
        except Exception as exc:
            print(f"Error loading {url}: {exc}")
    if not frames:
        raise RuntimeError(f"No historical data was downloaded for {config.key}")
    result = pd.concat(frames, ignore_index=True)
    output = Path(output_dir or os.getenv("PITCH_ORACLE_DATA_DIR", config.data_dir_name))
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "combined_historical_data.csv", sep="\t", index=False)
    return result


if __name__ == "__main__":
    combine_raw_data(os.getenv("PITCH_ORACLE_LEAGUE", "epl"))

