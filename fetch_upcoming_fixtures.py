"""Fetch upcoming fixtures from ESPN using a configured league slug."""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import os
import requests
import pandas as pd

from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.leagues import get_league_config
from team_name_mapping import normalize_team_name


def fetch_upcoming_fixtures(
    league: LeagueConfig | str = "epl", days_ahead: int = 60,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    config = get_league_config(league) if isinstance(league, str) else league
    if not config.espn_slug:
        raise ValueError(f"No ESPN slug configured for {config.key}")
    now = datetime.now().astimezone()
    date_range = f"{now:%Y%m%d}-{(now + timedelta(days=days_ahead)):%Y%m%d}"
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{config.espn_slug}/scoreboard?dates={date_range}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    rows = []
    for event in response.json().get("events", []):
        competition = (event.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c.get("team", {}).get("displayName", "") for c in competition.get("competitors", [])}
        status = event.get("status", {}).get("type", {}).get("name", "")
        if status in {"STATUS_FINAL", "STATUS_FULL_TIME"} or not teams.get("home") or not teams.get("away"):
            continue
        kickoff = datetime.fromisoformat(event["date"].replace("Z", "+00:00")).astimezone(ZoneInfo("US/Eastern"))
        rows.append({"Date": kickoff.strftime("%Y-%m-%d"), "Time": kickoff.strftime("%H:%M"),
                     "HomeTeam": normalize_team_name(teams["home"], config.team_aliases),
                     "AwayTeam": normalize_team_name(teams["away"], config.team_aliases), "Status": status})
    result = pd.DataFrame(rows, columns=["Date", "Time", "HomeTeam", "AwayTeam", "Status"])
    active_output = output_dir or os.getenv("PITCH_ORACLE_DATA_DIR", config.data_dir_name)
    if active_output:
        output = Path(active_output); output.mkdir(parents=True, exist_ok=True)
        result.to_csv(output / "upcoming_fixtures.csv", index=False)
    return result


if __name__ == "__main__":
    fetch_upcoming_fixtures(os.getenv("PITCH_ORACLE_LEAGUE", "epl"))

