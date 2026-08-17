"""Build chronological team-event features as an explicit, import-safe command."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import unicodedata
from zoneinfo import ZoneInfo

import pandas as pd

from pitch_oracle_core.domain.competitions import edition_from_league_config
from pitch_oracle_core.features import completed_match_rows
from pitch_oracle_core.features.ledger import (
    add_prior_team_state,
    build_team_events,
    match_feature_snapshots,
)
from pitch_oracle_core.leagues import get_league_config
from pitch_oracle_core.pipelines import atomic_output


COLUMN_RENAMES = {
    "Div": "Division", "Date": "MatchDate", "Time": "KickoffTime",
    "FTHG": "FullTimeHomeGoals", "FTAG": "FullTimeAwayGoals",
    "FTR": "FullTimeResult", "HTHG": "HalfTimeHomeGoals",
    "HTAG": "HalfTimeAwayGoals", "HTR": "HalfTimeResult",
    "HS": "HomeShots", "AS": "AwayShots", "HST": "HomeShotsOnTarget",
    "AST": "AwayShotsOnTarget", "HF": "HomeFouls", "AF": "AwayFouls",
    "HC": "HomeCorners", "AC": "AwayCorners", "HY": "HomeYellowCards",
    "AY": "AwayYellowCards", "HR": "HomeRedCards", "AR": "AwayRedCards",
}


def _slug(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")


def _kickoff_utc(row: pd.Series, local_timezone: str) -> pd.Timestamp:
    date_value = pd.Timestamp(row["MatchDate"]).strftime("%Y-%m-%d")
    time_value = row.get("KickoffTime", "12:00")
    if pd.isna(time_value):
        time_value = "12:00"
    value = f"{date_value} {time_value}"
    parsed = pd.to_datetime(value, format="%Y-%m-%d %H:%M", errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid match kickoff: {value!r}")
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(ZoneInfo(local_timezone))
    return timestamp.tz_convert("UTC")


def prepare_historical_features(
    *,
    league_key: str,
    source: str | Path,
    destination: str | Path,
    xg_source: str | Path | None = None,
) -> pd.DataFrame:
    config = get_league_config(league_key)
    source = Path(source)
    frame = pd.read_csv(source, sep="\t").rename(columns=COLUMN_RENAMES)
    required = {
        "MatchDate", "HomeTeam", "AwayTeam", "FullTimeHomeGoals",
        "FullTimeAwayGoals", "FullTimeResult",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Historical source misses: {sorted(missing)}")
    frame = completed_match_rows(frame, result_column="FullTimeResult").copy()
    if xg_source is not None:
        xg_path = Path(xg_source)
        if xg_path.exists():
            xg_frame = pd.read_csv(xg_path, sep="\t")
            for column in ("HomeTeam", "AwayTeam", "match_date", "home_xg", "away_xg"):
                if column not in xg_frame:
                    raise ValueError(f"xG source {xg_path} misses {column!r}")
            frame["MatchDate"] = pd.to_datetime(frame["MatchDate"]).dt.strftime("%Y-%m-%d")
            xg_frame["match_date"] = pd.to_datetime(xg_frame["match_date"]).dt.strftime("%Y-%m-%d")
            frame = frame.merge(
                xg_frame[["HomeTeam", "AwayTeam", "match_date", "home_xg", "away_xg"]],
                how="left",
                left_on=["HomeTeam", "AwayTeam", "MatchDate"],
                right_on=["HomeTeam", "AwayTeam", "match_date"],
            ).drop(columns=["match_date"])
    frame["kickoff_utc"] = frame.apply(
        _kickoff_utc, axis=1, local_timezone=config.sources.weather_timezone
    )
    frame = frame.sort_values(["kickoff_utc", "HomeTeam", "AwayTeam"], kind="stable")
    frame["fixture_id"] = [
        f"{league_key}:{timestamp:%Y%m%dT%H%MZ}:{_slug(home)}:{_slug(away)}:{index}"
        for index, (timestamp, home, away) in enumerate(
            zip(frame.kickoff_utc, frame.HomeTeam, frame.AwayTeam), start=1
        )
    ]
    frame["home_team_id"] = frame.HomeTeam.map(
        lambda value: f"{league_key}:{_slug(config.team_aliases.get(str(value), str(value)))}"
    )
    frame["away_team_id"] = frame.AwayTeam.map(
        lambda value: f"{league_key}:{_slug(config.team_aliases.get(str(value), str(value)))}"
    )
    frame["edition_id"] = frame.kickoff_utc.map(
        lambda kickoff: edition_from_league_config(
            config,
            kickoff.astimezone(ZoneInfo(config.sources.weather_timezone)).year
            if kickoff.astimezone(ZoneInfo(config.sources.weather_timezone)).month
            >= config.season_months[0]
            else kickoff.astimezone(ZoneInfo(config.sources.weather_timezone)).year - 1,
        ).edition_id
    )
    frame["rules_version"] = frame.edition_id.map(
        lambda edition: f"{edition}-rules-v1"
    )
    matches = frame.rename(columns={
        "FullTimeHomeGoals": "home_goals",
        "FullTimeAwayGoals": "away_goals",
        "HomeShots": "home_shots", "AwayShots": "away_shots",
        "HomeShotsOnTarget": "home_shots_on_target",
        "AwayShotsOnTarget": "away_shots_on_target",
    })
    events = add_prior_team_state(build_team_events(matches))
    snapshots = match_feature_snapshots(events)
    result = frame.merge(snapshots, on="fixture_id", how="left", validate="one_to_one")
    result["Season"] = result.edition_id.str.rsplit(":", n=1).str[-1]
    legacy_aliases = {
        "home_points_l5": "HomeTeamPointsLast5",
        "away_points_l5": "AwayTeamPointsLast5",
        "home_rest_days": "HomeRestDays",
        "away_rest_days": "AwayRestDays",
        "home_goals_for_l5": "HomeGoalsAve",
        "away_goals_for_l5": "AwayGoalsAve",
    }
    for canonical, legacy in legacy_aliases.items():
        if canonical in result:
            result[legacy] = result[canonical]
    result["feature_timestamp"] = result["kickoff_utc"] - pd.Timedelta(microseconds=1)
    destination = Path(destination)
    atomic_output(
        destination,
        lambda temporary: result.to_csv(temporary, sep="\t", index=False),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default=os.getenv("PITCH_ORACLE_LEAGUE", "epl"))
    parser.add_argument("--data-dir", default=os.getenv("PITCH_ORACLE_DATA_DIR", "data_files"))
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    result = prepare_historical_features(
        league_key=args.league,
        source=data_dir / "combined_historical_data.csv",
        destination=data_dir / "combined_historical_data_with_calculations_new.csv",
        xg_source=data_dir / "pitchapi_match_xg.csv",
    )
    print(f"Wrote {len(result)} chronological feature rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
