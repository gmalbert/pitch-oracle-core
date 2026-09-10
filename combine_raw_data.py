"""Download and combine football-data.co.uk history for any configured league.

Powered by ``penaltyblog.scrapers.FootballData`` for URL construction, season
mapping, and HTTP fetching.  The raw CSV is parsed locally to preserve the
original column casing required by ``prepare_model_data.py``'s ``COLUMN_RENAMES``.
"""

from datetime import date
from pathlib import Path
from typing import Iterable
import io
import os
import time
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests

from penaltyblog.scrapers import FootballData

from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.features import parse_match_dates
from pitch_oracle_core.leagues import get_league_config
from pitch_oracle_core.team_mappings import penaltyblog_competition


FOOTBALL_DATA_HOSTS = ("football-data.co.uk", "www.football-data.co.uk")
FOOTBALL_DATA_REQUIRED_COLUMNS = {
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
}
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 30


def _host_variant(url: str, host: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


def _retry_delay(response: requests.Response) -> float:
    value = response.headers.get("Retry-After", "")
    try:
        return min(max(float(value), 0.0), 5.0)
    except ValueError:
        return 0.5


def _decode_csv(response: requests.Response) -> str:
    payload = response.content
    if payload.startswith(b"\xef\xbb\xbf"):
        payload = payload[3:]
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("cp1252")


def _download_football_data(url: str) -> pd.DataFrame:
    """Download one Football-Data CSV with host fallback and validation."""
    errors = []
    for host in FOOTBALL_DATA_HOSTS:
        candidate = _host_variant(url, host)
        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                response = requests.get(
                    candidate,
                    headers={"Accept": "text/csv,text/plain;q=0.9,*/*;q=0.1"},
                    timeout=DOWNLOAD_TIMEOUT_SECONDS,
                )
                if 500 <= response.status_code < 600 and attempt < DOWNLOAD_ATTEMPTS - 1:
                    time.sleep(_retry_delay(response))
                    continue
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                preview = response.text[:160].replace("\n", " ").strip()
                if "<html" in preview.lower() or "<!doctype" in preview.lower():
                    raise ValueError(
                        f"HTML response (content-type={content_type or 'unknown'}, preview={preview!r})"
                    )
                frame = pd.read_csv(io.StringIO(_decode_csv(response)))
                missing = FOOTBALL_DATA_REQUIRED_COLUMNS.difference(frame.columns)
                if missing:
                    raise ValueError(
                        f"missing required columns {sorted(missing)} "
                        f"(content-type={content_type or 'unknown'}, preview={preview!r})"
                    )
                return frame
            except (requests.RequestException, UnicodeDecodeError, ValueError) as exc:
                errors.append(f"{candidate} [attempt {attempt + 1}]: {exc}")
                if isinstance(exc, requests.RequestException) and attempt < DOWNLOAD_ATTEMPTS - 1:
                    if getattr(exc.response, "status_code", 0) >= 500:
                        time.sleep(_retry_delay(exc.response))
                        continue
                break
    raise RuntimeError("Football-Data download failed: " + " | ".join(errors))


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
    competition = penaltyblog_competition(config, "footballdata")
    frames = []
    errors = []
    for season in seasons:
        season_label = f"20{season[:2]}-20{season[2:]}"
        url = "unknown"
        try:
            fd = FootballData(competition, season_label)
            url = fd.base_url.format(
                season=fd.mapped_season, competition=fd.mapped_competition,
            )
            frame = _download_football_data(url)
            parsed_dates = parse_match_dates(frame["Date"])
            if parsed_dates.isna().any():
                raise ValueError(f"Downloaded season {season} contains invalid match dates")
            frame["Date"] = parsed_dates.dt.strftime("%Y-%m-%d")
            frame["League"] = config.key
            frames.append(frame)
        except Exception as exc:
            message = f"{season} ({url}): {exc}"
            errors.append(message)
            print(f"Error loading season {season} for {competition}: {exc}")
        time.sleep(0.5)   # polite rate limit
    if not frames:
        details = "; ".join(errors) if errors else "no seasons were requested"
        raise RuntimeError(
            f"No historical data was downloaded for {config.key}. Details: {details}"
        )
    result = pd.concat(frames, ignore_index=True)
    output = Path(output_dir or os.getenv("PITCH_ORACLE_DATA_DIR", config.data_dir_name))
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "combined_historical_data.csv", sep="\t", index=False)
    return result


if __name__ == "__main__":
    combine_raw_data(os.getenv("PITCH_ORACLE_LEAGUE", "epl"))
