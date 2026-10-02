"""Goalkeeper values with exposure, non-own goals, and strong sample shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .lineup_strength import shrink_player


def keeper_match_value(frame: pd.DataFrame) -> pd.Series:
    xgot = pd.to_numeric(frame["xgot_faced"], errors="coerce")
    goals = pd.to_numeric(frame["goals_conceded_non_own_goal"], errors="coerce")
    return xgot - goals


def keeper_state(matches: pd.DataFrame, *, target_kickoff: pd.Timestamp, as_of: pd.Timestamp | None = None, span: int = 10, prior_minutes: float = 1350.0) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of if as_of is not None else target_kickoff - pd.Timedelta(microseconds=1))
    if cutoff.tzinfo is None or pd.Timestamp(target_kickoff).tzinfo is None or cutoff >= target_kickoff:
        raise ValueError("Keeper state requires a pre-kickoff timezone-aware cutoff")
    if span < 1 or prior_minutes <= 0:
        raise ValueError("Keeper span and prior must be positive")
    if matches.empty:
        return pd.DataFrame(columns=["player_id", "keeper_strength", "effective_minutes", "shots_on_target_faced", "sample_status", "source_observed_at"])
    required = {"fixture_id", "player_id", "observed_at", "minutes", "xgot_faced", "goals_conceded_non_own_goal", "shots_on_target_faced"}
    if required.difference(matches):
        raise ValueError(f"Keeper history misses {sorted(required.difference(matches))}")
    frame = matches.copy()
    frame["observed_at"] = pd.to_datetime(frame.observed_at, utc=True, errors="raise")
    if frame.observed_at.isna().any():
        raise ValueError("Keeper history requires observation lineage")
    frame = frame.loc[frame.observed_at <= cutoff].sort_values("observed_at").drop_duplicates(["fixture_id", "player_id"], keep="last")
    if "source_kickoff_utc" in frame:
        frame["source_kickoff_utc"] = pd.to_datetime(frame.source_kickoff_utc, utc=True, errors="raise")
        if frame.source_kickoff_utc.isna().any() or (frame.source_kickoff_utc >= cutoff).any():
            raise ValueError("Keeper source fixture must predate the forecast cutoff")
        frame = frame.sort_values(["source_kickoff_utc", "fixture_id"])
    frame["xgot_prevented"] = keeper_match_value(frame)
    rows = []
    for player_id, history in frame.groupby("player_id", sort=True):
        history = history.tail(span)
        minutes = pd.to_numeric(history.minutes, errors="coerce")
        shots = pd.to_numeric(history.shots_on_target_faced, errors="coerce")
        if (minutes.dropna() < 0).any() or (shots.dropna() < 0).any():
            raise ValueError("Keeper exposure cannot be negative")
        valid = minutes.gt(0) & history.xgot_prevented.notna() & shots.notna()
        sample_minutes = float(minutes.loc[valid].sum())
        exposure = float(shots.loc[valid].sum())
        raw_rate = float(history.loc[valid, "xgot_prevented"].sum()) * 90 / sample_minutes if sample_minutes else 0.0
        sample = "strong_prior" if sample_minutes < 450 else "moderate_prior" if sample_minutes < 1350 else "normal"
        row = {"player_id": player_id, "keeper_strength": shrink_player(raw_rate, sample_minutes, prior_minutes), "effective_minutes": sample_minutes, "shots_on_target_faced": exposure, "sample_status": sample, "source_observed_at": history.observed_at.max().isoformat(), "keeper_xgot_prevented_per_shot": float(history.loc[valid, "xgot_prevented"].sum()) / exposure if exposure > 0 else np.nan}
        for numerator, denominator, label in (("keeper_claims_won", "keeper_claims", "keeper_claim_rate"), ("keeper_distributions_completed", "keeper_distributions", "keeper_distribution_accuracy")):
            if numerator in history and denominator in history:
                a, b = pd.to_numeric(history[numerator], errors="coerce"), pd.to_numeric(history[denominator], errors="coerce")
                known = a.notna() & b.gt(0)
                row[label] = float(a.loc[known].sum() / b.loc[known].sum()) if known.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
