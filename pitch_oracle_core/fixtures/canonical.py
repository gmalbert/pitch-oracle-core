"""Shared stable canonical identities for historical and provider-crosscheck inputs."""

from __future__ import annotations

from zoneinfo import ZoneInfo
import pandas as pd

from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.data.normalization import stable_fixture_id
from pitch_oracle_core.domain.competitions import edition_from_league_config
from pitch_oracle_core.domain.entities import normalized_name
from pitch_oracle_core.features import parse_match_dates


def canonical_fixture_frame(source: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
    frame = source.rename(columns={"Date": "MatchDate", "Time": "KickoffTime"}).copy()
    required = {"HomeTeam", "AwayTeam", "MatchDate"}
    if required.difference(frame):
        raise ValueError(f"Canonical source misses {sorted(required.difference(frame))}")
    dates = parse_match_dates(frame.MatchDate)
    if dates.isna().any():
        raise ValueError("Canonical fixtures require valid dates")
    rows = []
    for (index, row), date in zip(frame.iterrows(), dates):
        raw_kickoff = row.get("kickoff_utc")
        supplied_time = row.get("KickoffTime")
        if pd.notna(raw_kickoff):
            kickoff = pd.Timestamp(raw_kickoff)
            if kickoff.tzinfo is None:
                raise ValueError("Canonical kickoff must be timezone-aware")
            precision = row.get("kickoff_precision", "exact")
        else:
            clock = str(supplied_time) if pd.notna(supplied_time) else "12:00"
            kickoff = pd.Timestamp(f"{date:%Y-%m-%d} {clock}").tz_localize(ZoneInfo(config.sources.weather_timezone))
            precision = "exact" if pd.notna(supplied_time) else "date_only"
        kickoff = kickoff.tz_convert("UTC")
        local = kickoff.tz_convert(ZoneInfo(config.sources.weather_timezone))
        year = local.year if local.month >= config.season_months[0] else local.year - 1
        edition = edition_from_league_config(config, year)
        names = {}
        for side, field in (("home", "HomeTeam"), ("away", "AwayTeam")):
            canonical_name = config.team_aliases.get(str(row[field]), str(row[field]))
            internal = row.get(f"{side}_team_id")
            names[f"{side}_team_id"] = str(internal) if pd.notna(internal) else f"{config.key}:{normalized_name(canonical_name).replace(' ', '-')}"
            names[f"{side}_display_name"] = canonical_name
        if names["home_team_id"] == names["away_team_id"]:
            raise ValueError("Canonical teams must be distinct")
        identity = f"{kickoff.isoformat()}|{names['home_team_id']}|{names['away_team_id']}"
        existing_id = row.get("fixture_id")
        lower_bound = kickoff if precision == "exact" else local.normalize().tz_convert("UTC")
        rows.append({**row.to_dict(), **names, "fixture_id": str(existing_id) if pd.notna(existing_id) else stable_fixture_id(edition.edition_id, "canonical", identity), "league_key": config.key, "edition_id": row.get("edition_id") if pd.notna(row.get("edition_id")) else edition.edition_id, "rules_version": edition.rules_version, "kickoff_utc": kickoff, "kickoff_precision": precision, "kickoff_lower_bound_utc": lower_bound, "MatchDate": date.strftime("%Y-%m-%d"), "status": row.get("status", "finished" if pd.notna(row.get("FTR", row.get("FullTimeResult"))) else "scheduled")})
    result = pd.DataFrame(rows)
    if not result.empty and result.fixture_id.duplicated().any():
        raise ValueError("Duplicate canonical fixture identity")
    return result
