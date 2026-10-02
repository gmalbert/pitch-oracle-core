"""Per-team snapshot selection and append-only lineup observation storage."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from pitch_oracle_core.pitchapi.storage import append_revisions
from pitch_oracle_core.pitchapi.normalize import boolean

LINEUP_PRIORITY = {"unknown": 0, "predicted": 1, "confirmed": 2}


def store_lineup_snapshots(frame: pd.DataFrame, destination: str | Path) -> pd.DataFrame:
    return append_revisions(frame, destination, keys=["snapshot_id", "player_id"])


def latest_eligible_lineup(
    snapshots: pd.DataFrame, *, fixture_id: str, as_of: pd.Timestamp,
    team_id: str | None = None, require_complete: bool = False,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("Lineup cutoff must be timezone-aware")
    if snapshots.empty:
        return snapshots.copy()
    required = {"fixture_id", "team_id", "player_id", "snapshot_at", "kickoff_utc", "lineup_status", "snapshot_id", "is_starter"}
    if required.difference(snapshots):
        raise ValueError(f"Lineup history misses {sorted(required.difference(snapshots))}")
    frame = snapshots.loc[snapshots.fixture_id.astype(str) == str(fixture_id)].copy()
    if team_id is not None:
        frame = frame.loc[frame.team_id.astype(str) == str(team_id)]
    frame["snapshot_at"] = pd.to_datetime(frame.snapshot_at, utc=True, errors="raise")
    frame["kickoff_utc"] = pd.to_datetime(frame.kickoff_utc, utc=True, errors="raise")
    frame = frame.loc[(frame.snapshot_at <= cutoff) & (frame.snapshot_at < frame.kickoff_utc)]
    if frame.empty:
        return frame
    frame["_priority"] = frame.lineup_status.map(LINEUP_PRIORITY).fillna(0)
    selected = []
    for _, team in frame.groupby("team_id", sort=True):
        # A later predicted response is audited as a reversion; it does not erase an official XI.
        priority = team._priority.max()
        at_stage = team.loc[team._priority == priority]
        newest = at_stage.loc[at_stage.snapshot_at == at_stage.snapshot_at.max()]
        if newest.snapshot_id.nunique() != 1:
            raise ValueError("Ambiguous lineup observations at the same team timestamp")
        if newest.player_id.duplicated().any():
            raise ValueError("Duplicate player within a lineup snapshot")
        starters = newest.loc[newest.is_starter.map(boolean).eq(True)]
        if require_complete and (len(starters) != 11 or priority == 0):
            continue
        selected.append(newest)
    return pd.concat(selected, ignore_index=True).drop(columns="_priority") if selected else frame.iloc[:0].drop(columns="_priority")


def lineup_status(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "none"
    statuses = set(frame.lineup_status.astype(str))
    starters = frame.loc[frame.is_starter.map(boolean).eq(True)]
    if len(starters) != 11 or len(statuses) != 1:
        return "incomplete"
    return next(iter(statuses))


def lineup_continuity(current: set[str], previous: set[str]) -> float:
    if len(current) != 11 or len(previous) != 11:
        return float("nan")
    return len(current.intersection(previous)) / 11.0
