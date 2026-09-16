"""Fetch upcoming fixtures from ESPN using a configured league slug."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from zoneinfo import ZoneInfo
import os
import requests
import pandas as pd

from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.leagues import get_league_config
from pitch_oracle_core.domain.competitions import edition_from_league_config
from pitch_oracle_core.domain.entities import normalized_name
from pitch_oracle_core.team_mappings import normalize_team_name


def _scoreboard_dates(now: datetime, days_ahead: int) -> list[str]:
    if days_ahead < 0:
        raise ValueError("days_ahead must be non-negative")
    return [
        f"{now + timedelta(days=offset):%Y%m%d}"
        for offset in range(days_ahead + 1)
    ]


def _scoreboard_events(
    config: LeagueConfig, now: datetime, days_ahead: int,
    fallback_delay_seconds: float,
) -> list[dict]:
    fixture_dates = _scoreboard_dates(now, days_ahead)
    base_url = (
        "https://site.api.espn.com/apis/site/v2/sports/soccer/"
        f"{config.espn_slug}/scoreboard"
    )
    date_range = fixture_dates[0]
    if len(fixture_dates) > 1:
        date_range = f"{fixture_dates[0]}-{fixture_dates[-1]}"
    try:
        response = requests.get(
            base_url, params={"limit": 1000, "dates": date_range}, timeout=15
        )
        response.raise_for_status()
        return response.json().get("events", [])
    except requests.HTTPError as exc:
        if getattr(exc.response, "status_code", None) != 400 or len(fixture_dates) == 1:
            raise
    events = []
    for offset, fixture_date in enumerate(fixture_dates):
        if offset and fallback_delay_seconds:
            sleep(fallback_delay_seconds)
        response = requests.get(
            base_url, params={"dates": fixture_date}, timeout=15
        )
        response.raise_for_status()
        events.extend(response.json().get("events", []))
    return events


def fetch_upcoming_fixtures(
    league: LeagueConfig | str = "epl", days_ahead: int = 60,
    output_dir: str | Path | None = None, fallback_delay_seconds: float = 1.0,
) -> pd.DataFrame:
    config = get_league_config(league) if isinstance(league, str) else league
    if not config.espn_slug:
        raise ValueError(f"No ESPN slug configured for {config.key}")
    now = datetime.now(timezone.utc)
    events = _scoreboard_events(config, now, days_ahead, fallback_delay_seconds)
    rows = []
    seen_event_ids = set()
    for event in events:
        event_id = event.get("id")
        if event_id is None or event_id in seen_event_ids or not event.get("date"):
            continue
        seen_event_ids.add(event_id)
        competition = (event.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c.get("team", {}).get("displayName", "") for c in competition.get("competitors", [])}
        status = event.get("status", {}).get("type", {}).get("name", "")
        if status in {"STATUS_FINAL", "STATUS_FULL_TIME"} or not teams.get("home") or not teams.get("away"):
            continue
        kickoff_utc = datetime.fromisoformat(
            event["date"].replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        local = kickoff_utc.astimezone(ZoneInfo(config.sources.weather_timezone))
        season_start = local.year if local.month >= config.season_months[0] else local.year - 1
        edition = edition_from_league_config(config, season_start)
        home_name = normalize_team_name(teams["home"], config.team_aliases)
        away_name = normalize_team_name(teams["away"], config.team_aliases)
        rows.append({
            "fixture_id": f"{config.key}:espn:{event_id}",
            "provider_event_id": str(event_id), "source": "espn",
            "observed_at": now.isoformat(), "kickoff_utc": kickoff_utc.isoformat(),
            "edition_id": edition.edition_id, "rules_version": edition.rules_version,
            "Date": local.strftime("%Y-%m-%d"), "Time": local.strftime("%H:%M"),
            "HomeTeam": home_name, "AwayTeam": away_name,
            "home_team_id": f"{config.key}:{normalized_name(home_name).replace(' ', '-')}",
            "away_team_id": f"{config.key}:{normalized_name(away_name).replace(' ', '-')}",
            "Status": status,
        })
    result = pd.DataFrame(rows, columns=[
        "fixture_id", "provider_event_id", "source", "observed_at", "kickoff_utc",
        "edition_id", "rules_version", "Date", "Time", "HomeTeam", "AwayTeam",
        "home_team_id", "away_team_id", "Status",
    ])
    active_output = output_dir or os.getenv("PITCH_ORACLE_DATA_DIR", config.data_dir_name)
    if active_output:
        output = Path(active_output); output.mkdir(parents=True, exist_ok=True)
        result.to_csv(output / "upcoming_fixtures.csv", index=False)
    return result


if __name__ == "__main__":
    fetch_upcoming_fixtures(os.getenv("PITCH_ORACLE_LEAGUE", "epl"))

