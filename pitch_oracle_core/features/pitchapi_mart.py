"""Build PitchAPI features once, with identical semantics for training and serving."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from pitch_oracle_core.pitchapi.storage import read_frame
from pitch_oracle_core.players.goalkeeper import build_keeper_matches
from pitch_oracle_core.players.squad_features import build_squad_features
from .advanced_team import METRICS, build_advanced_team_features, shot_summaries_to_team_observations
from .style_matchups import add_style_matchups


def xg_observations(xg: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Legacy unversioned xG remains descriptive and cannot populate replay state."""
    columns = ["fixture_id", "team_id", "observed_at", "xg"]
    if xg.empty or "observed_at" not in xg:
        return pd.DataFrame(columns=columns)
    xg = xg.copy()
    if "fixture_id" not in xg:
        required = {"HomeTeam", "AwayTeam", "match_date"}
        if required.difference(xg):
            raise ValueError("xG observations require canonical identities or exact team/date keys")
        xg["match_date"] = pd.to_datetime(xg.match_date).dt.strftime("%Y-%m-%d")
        lookup = fixtures[["fixture_id", "HomeTeam", "AwayTeam", "MatchDate"]].copy()
        lookup["match_date"] = pd.to_datetime(lookup.pop("MatchDate")).dt.strftime("%Y-%m-%d")
        xg = xg.merge(lookup, on=["HomeTeam", "AwayTeam", "match_date"], validate="many_to_one")
    return shot_summaries_to_team_observations(xg[["fixture_id", "observed_at", "home_xg", "away_xg"]], fixtures).rename(columns={"xg": "xg"})


def build_pitchapi_feature_mart(targets: pd.DataFrame, source_fixtures: pd.DataFrame, *, data_dir: str | Path | None = None, summaries: pd.DataFrame | None = None, advanced: pd.DataFrame | None = None, players: pd.DataFrame | None = None, lineups: pd.DataFrame | None = None, shots: pd.DataFrame | None = None, xg: pd.DataFrame | None = None, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Actual observation times are mandatory; a backfill never rewrites chronology."""
    def source(frame, name):
        return frame if frame is not None else read_frame(Path(data_dir) / name) if data_dir is not None else pd.DataFrame()

    summaries = source(summaries, "pitchapi_match_shot_features.csv")
    advanced = source(advanced, "pitchapi_advanced_team.parquet")
    players = source(players, "pitchapi_player_match.parquet")
    lineups = source(lineups, "pitchapi_lineup_snapshots.parquet")
    shots = source(shots, "pitchapi_shots.parquet")
    xg = source(xg, "pitchapi_match_xg.csv")
    result = targets[["fixture_id"]].copy()
    lineage = []
    shot_history = shot_summaries_to_team_observations(summaries, source_fixtures) if not summaries.empty else xg_observations(xg, source_fixtures)
    for history, families in ((shot_history, ("xg", "shot_profile")), (advanced, ("advanced_team",))):
        metrics = tuple(metric for metric in METRICS if metric.family in families)
        built = build_advanced_team_features(targets, history, source_fixtures, as_of=as_of, metrics=metrics)
        lineage.append(pd.to_datetime(built.pop("feature_observed_at"), utc=True))
        if "as_of" in result:
            built = built.drop(columns="as_of")
        result = result.merge(built, on="fixture_id", validate="one_to_one")
    response_revisions = read_frame(Path(data_dir) / "pitchapi_response_revisions.parquet") if data_dir is not None else pd.DataFrame()
    keepers = build_keeper_matches(players, shots, source_fixtures, response_revisions=response_revisions)
    squad = build_squad_features(targets, players, lineups, source_fixtures, keeper_matches=keepers, as_of=as_of)
    lineage.append(pd.to_datetime(squad.pop("squad_feature_observed_at"), utc=True))
    result = result.merge(squad, on="fixture_id", validate="one_to_one")
    result = add_style_matchups(result)
    result["feature_observed_at"] = pd.concat(lineage, axis=1).max(axis=1).map(lambda value: value.isoformat() if pd.notna(value) else None)
    for side in ("home", "away"):
        result[f"{side}_shot_quality_ewm10"] = result.get(f"{side}_xg_per_shot_ewm10", np.nan)
    return result


def attach_pitchapi_features(targets: pd.DataFrame, source_fixtures: pd.DataFrame, **kwargs) -> pd.DataFrame:
    mart = build_pitchapi_feature_mart(targets, source_fixtures, **kwargs)
    return targets.drop(columns=[column for column in mart if column != "fixture_id" and column in targets]).merge(mart, on="fixture_id", how="left", validate="one_to_one")
