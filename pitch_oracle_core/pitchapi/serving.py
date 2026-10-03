"""Validated current forecasts for consumers that retain a legacy prediction log."""

from pathlib import Path

import pandas as pd

from pitch_oracle_core.features.families import FeatureFamilyConfig
from .artifacts import analytics_repository


def legacy_prediction_overlay(log: pd.DataFrame, *, league_key: str, root: str | Path = ".", now=None) -> pd.DataFrame:
    try:
        return _overlay(log, league_key=league_key, root=root, now=now)
    except (OSError, ValueError, KeyError, TypeError):
        return log


def _overlay(log: pd.DataFrame, *, league_key: str, root: str | Path, now) -> pd.DataFrame:
    """Use recent, integrity-checked promoted forecasts without rewriting the log."""
    config = FeatureFamilyConfig.load(Path(root) / "data_files/pitchapi_feature_config.json", league_key=league_key)
    if not config.enabled_families:
        return log
    repository = analytics_repository(root, league_key)
    if not repository.available("pitchapi_upcoming_predictions"):
        return log
    frame = repository.frame("pitchapi_upcoming_predictions")
    if frame.empty:
        return log
    cutoff = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    issued = pd.to_datetime(frame.PredictionDate, utc=True, errors="raise")
    kickoff = pd.to_datetime(frame.kickoff_utc, utc=True, errors="raise")
    frame = frame.loc[(issued <= cutoff) & (issued >= cutoff - pd.Timedelta(hours=2)) & (kickoff > cutoff) & frame.evidence_id.eq(config.evidence_id)].copy()
    if frame.empty:
        return log
    if log.empty:
        return frame
    keys = ["MatchDate", "HomeTeam", "AwayTeam"]
    old = log.copy()
    old["MatchDate"] = pd.to_datetime(old.MatchDate, errors="raise").dt.strftime("%Y-%m-%d")
    current = set(map(tuple, frame[keys].to_numpy()))
    replaced = old[keys].apply(tuple, axis=1).isin(current) & old.ModelVersion.isin(["ensemble_v1", "pitchapi_v1"]) & old.ActualResult.isna()
    return pd.concat([old.loc[~replaced], frame], ignore_index=True)
