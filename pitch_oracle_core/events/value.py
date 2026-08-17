"""Lagged provider-neutral xT/VAEP-style action-value snapshots."""

import numpy as np
import pandas as pd


def action_value(start_value: float, end_value: float, successful: bool | None) -> float:
    if successful is False:
        return -max(0.0, start_value)
    return float(end_value - start_value)


def lagged_action_snapshots(actions: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    frame = actions.copy()
    fixture_times = fixtures[["fixture_id", "kickoff_utc"]].copy()
    fixture_times["kickoff_utc"] = pd.to_datetime(fixture_times["kickoff_utc"], utc=True)
    frame = frame.merge(fixture_times, on="fixture_id", validate="many_to_one")
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    if (frame.observed_at < frame.kickoff_utc).any():
        # In-match events are observed after their own kickoff. The guard prevents
        # accidentally treating them as pre-match features for that same fixture.
        raise ValueError("event observation predates its source fixture kickoff")
    summary = frame.groupby(["fixture_id", "team_id"], as_index=False).agg(
        action_value=("action_value", "sum"), actions=("action_id", "nunique"),
        observed_at=("observed_at", "max"),
    )
    return summary.sort_values(["team_id", "observed_at"])


def action_features_at_kickoff(
    action_summaries: pd.DataFrame,
    target_fixtures: pd.DataFrame,
    *,
    half_life_days: float = 90.0,
) -> pd.DataFrame:
    """Attach recency-weighted prior action value without same-fixture leakage."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    required = {"fixture_id", "team_id", "action_value", "actions", "observed_at"}
    missing = required.difference(action_summaries.columns)
    if missing:
        raise ValueError(f"Missing action summary columns: {sorted(missing)}")
    summaries = action_summaries.copy()
    summaries["observed_at"] = pd.to_datetime(summaries["observed_at"], utc=True)
    fixtures = target_fixtures.copy()
    fixtures["kickoff_utc"] = pd.to_datetime(fixtures["kickoff_utc"], utc=True)
    rows = []
    for fixture in fixtures.itertuples():
        for side in ("home", "away"):
            team_id = getattr(fixture, f"{side}_team_id")
            history = summaries.loc[
                (summaries.team_id == team_id)
                & (summaries.observed_at < fixture.kickoff_utc)
                & (summaries.fixture_id != fixture.fixture_id)
            ].copy()
            if history.empty:
                rows.append({
                    "fixture_id": fixture.fixture_id, "team_id": team_id,
                    "side": side, "lagged_action_value_per_action": np.nan,
                    "history_actions": 0, "feature_timestamp": pd.NaT,
                })
                continue
            age = (
                fixture.kickoff_utc - history.observed_at
            ).dt.total_seconds().to_numpy() / 86_400
            weight = np.exp(-np.log(2) * age / half_life_days)
            action_weight = weight * history.actions.to_numpy(dtype=float)
            denominator = action_weight.sum()
            value = (
                float(np.sum(weight * history.action_value.to_numpy(dtype=float)) / denominator)
                if denominator > 0 else np.nan
            )
            rows.append({
                "fixture_id": fixture.fixture_id, "team_id": team_id,
                "side": side, "lagged_action_value_per_action": value,
                "history_actions": int(history.actions.sum()),
                "feature_timestamp": history.observed_at.max(),
            })
    return pd.DataFrame(rows)
