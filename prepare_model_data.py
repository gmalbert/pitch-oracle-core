"""Build chronological team-event features as an explicit, import-safe command."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from pitch_oracle_core.features import completed_match_rows
from pitch_oracle_core.features.ledger import (
    add_prior_team_state,
    build_team_events,
    match_feature_snapshots,
)
from pitch_oracle_core.fixtures.canonical import canonical_fixture_frame
from pitch_oracle_core.features.pitchapi_mart import attach_pitchapi_features
from pitch_oracle_core.pitchapi.storage import read_frame
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


def prepare_historical_features(
    *,
    league_key: str,
    source: str | Path,
    destination: str | Path,
    xg_source: str | Path | None = None,
    pitchapi_data_dir: str | Path | None = None,
) -> pd.DataFrame:
    config = get_league_config(league_key)
    source = Path(source)
    frame = read_frame(source).rename(columns=COLUMN_RENAMES)
    required = {
        "MatchDate", "HomeTeam", "AwayTeam", "FullTimeHomeGoals",
        "FullTimeAwayGoals", "FullTimeResult",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Historical source misses: {sorted(missing)}")
    frame = completed_match_rows(frame, result_column="FullTimeResult").copy()
    frame = canonical_fixture_frame(frame, config)
    xg_frame = pd.DataFrame()
    if xg_source is not None:
        xg_path = Path(xg_source)
        if xg_path.exists():
            xg_frame = read_frame(xg_path)
            for column in ("HomeTeam", "AwayTeam", "match_date", "home_xg", "away_xg"):
                if column not in xg_frame:
                    raise ValueError(f"xG source {xg_path} misses {column!r}")
            frame["MatchDate"] = pd.to_datetime(frame["MatchDate"]).dt.strftime("%Y-%m-%d")
            xg_frame["match_date"] = pd.to_datetime(xg_frame["match_date"]).dt.strftime("%Y-%m-%d")
            descriptive = xg_frame.sort_values("observed_at") if "observed_at" in xg_frame else xg_frame
            keys = ["HomeTeam", "AwayTeam", "match_date"]
            if "observed_at" not in xg_frame and descriptive.duplicated(keys).any():
                raise ValueError("Unversioned xG contains ambiguous duplicate fixtures")
            descriptive = descriptive.drop_duplicates(keys, keep="last")
            frame = frame.merge(
                descriptive[[*keys, "home_xg", "away_xg"]],
                how="left",
                left_on=["HomeTeam", "AwayTeam", "MatchDate"],
                right_on=["HomeTeam", "AwayTeam", "match_date"],
                validate="one_to_one",
            ).drop(columns=["match_date"])
    frame = frame.sort_values(["kickoff_utc", "HomeTeam", "AwayTeam"], kind="stable")
    matches = frame.rename(columns={
        "FullTimeHomeGoals": "home_goals",
        "FullTimeAwayGoals": "away_goals",
        "HomeShots": "home_shots", "AwayShots": "away_shots",
        "HomeShotsOnTarget": "home_shots_on_target",
        "AwayShotsOnTarget": "away_shots_on_target",
    })
    # Raw provider xG has its own observation-aware ledger. Never feed it into
    # the football-data shift, which would fabricate availability for backfills.
    events = add_prior_team_state(build_team_events(matches.drop(columns=["home_xg", "away_xg"], errors="ignore")))
    snapshots = match_feature_snapshots(events)
    result = frame.merge(snapshots, on="fixture_id", how="left", validate="one_to_one")
    if pitchapi_data_dir is not None or not xg_frame.empty:
        result = attach_pitchapi_features(result, frame, data_dir=pitchapi_data_dir, xg=xg_frame if not xg_frame.empty else None)
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
        pitchapi_data_dir=data_dir,
    )
    print(f"Wrote {len(result)} chronological feature rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
