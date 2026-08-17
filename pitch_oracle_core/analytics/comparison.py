"""Team comparison and recency-weighted head-to-head context."""

from __future__ import annotations

import numpy as np
import pandas as pd


def percentile_comparison(
    snapshots: pd.DataFrame, team_ids: tuple[str, str], metrics: list[str]
) -> pd.DataFrame:
    missing = set(metrics).difference(snapshots.columns)
    if missing:
        raise ValueError(f"Unknown comparison metrics: {sorted(missing)}")
    percentiles = snapshots[metrics].rank(pct=True, method="average")
    percentiles["team_id"] = snapshots["team_id"]
    result = percentiles.loc[percentiles.team_id.isin(team_ids)].copy()
    if set(result.team_id) != set(team_ids):
        raise KeyError("Both comparison teams must exist in the snapshot")
    return result


def head_to_head_context(
    events: pd.DataFrame,
    team_a: str,
    team_b: str,
    *,
    as_of: object,
    half_life_days: float = 365.0,
) -> dict[str, object]:
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    frame = events.copy()
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True)
    selected = frame.loc[
        (frame.team_id == team_a)
        & (frame.opponent_id == team_b)
        & (frame.kickoff_utc < cutoff)
        & frame.points.notna()
    ].copy()
    if selected.empty:
        return {"matches": 0, "quality": "insufficient", "weighted_points_per_match": None}
    age = (cutoff - selected.kickoff_utc).dt.total_seconds() / 86_400
    weight = np.exp(-np.log(2) * age / half_life_days)
    effective_n = float(weight.sum())
    quality = "high" if effective_n >= 3 else "low_relevance" if effective_n >= 1 else "insufficient"
    return {
        "matches": len(selected),
        "effective_matches": effective_n,
        "quality": quality,
        "weighted_points_per_match": float(np.average(selected.points, weights=weight)),
        "meetings": selected.sort_values("kickoff_utc", ascending=False),
    }
