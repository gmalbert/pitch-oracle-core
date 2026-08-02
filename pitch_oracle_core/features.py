"""Shared point-in-time feature policy for model training and inference."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Literal

import numpy as np
import pandas as pd


def completed_match_rows(
    frame: pd.DataFrame,
    *,
    date_column: str = "MatchDate",
    result_column: str = "FullTimeResult",
) -> pd.DataFrame:
    """Return dated, completed matches suitable for point-in-time features."""
    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="coerce")
    valid = result[date_column].notna() & result[result_column].isin(("H", "D", "A"))
    return result.loc[valid].reset_index(drop=True)


FEATURE_POLICY_VERSION = 2

# These fields are outcomes, in-match observations, or aggregates currently
# calculated using the full historical dataset.  Using them to predict rows in
# that same dataset inflates validation scores and does not match live inference.
_EXCLUDED_COLUMNS = {
    "target", "FullTimeResult", "FullTimeHomeGoals", "FullTimeAwayGoals",
    "HalfTimeResult", "HalfTimeHomeGoals", "HalfTimeAwayGoals",
    "HomeWin", "AwayWin", "Draw", "WinningTeam",
    "HalfTimeHomeWin", "HalfTimeAwayWin", "HalfTimeDraw",
    "HomePoints", "AwayPoints", "HomeTeamCumulativePoints", "AwayTeamCumulativePoints",
    "HomeShots", "AwayShots", "HomeShotsOnTarget", "AwayShotsOnTarget",
    "HomeHitWoodwork", "AwayHitWoodwork", "HomeCorners", "AwayCorners",
    "HomeFouls", "AwayFouls", "HomeOffsides", "AwayOffsides",
    "HomeYellowCards", "AwayYellowCards", "HomeRedCards", "AwayRedCards",
    "HomeBookingPoints", "AwayBookingPoints", "Attendance",
    "MatchDate", "KickoffTime", "Season", "Venue", "Referee",
    "HomeTeam", "AwayTeam", "Division",
    # Full-sample team aggregates emitted by prepare_model_data.py. These can be
    # restored once they are generated point-in-time for every historical row.
    "HomeGoalsAve", "AwayGoalsAve", "HomeGoalsTotal", "AwayGoalsTotal",
    "HomeGoalsHalfAve", "AwayGoalsHalfAve", "HomeGoalsHalfTotal", "AwayGoalsHalfTotal",
    "HomeShotsAve", "AwayShotsAve", "HomeShotsTotal", "AwayShotsTotal",
    "HomeShotsOnTargetAve", "AwayShotsOnTargetAve",
    "HomeFirstHalfDifferentialAve", "AwayFirstHalfDifferentialAve",
    "HomeGameDifferentialAve", "AwayGameDifferentialAve",
    "HomeFirstToSecondHalfGoalRatioAve", "AwayFirstToSecondHalfGoalRatioAve",
}


def is_prematch_feature(column: str) -> bool:
    """Return whether a column is safe and available before kickoff."""
    name = str(column)
    if name in _EXCLUDED_COLUMNS:
        return False
    if name.startswith("Ref"):
        # Current referee aggregates include the row being predicted.
        return False
    if name.startswith("API_") or name == "API_StandingsRankDiff":
        # Current API snapshots are not historically versioned.  Applying a
        # current/final table to older fixtures leaks future season knowledge.
        return False
    if "ZScore" in name:
        # Existing season z-scores use full-season transforms.  Re-enable only
        # after they are calculated from the table state at each kickoff.
        return False
    if "Closing" in name:
        # Closing prices are unavailable when forecasts are generated earlier.
        return False
    return True


def prematch_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model columns in stable source order under the feature policy."""
    return [column for column in frame.columns if is_prematch_feature(column)]


def prior_group_rolling(
    frame: pd.DataFrame,
    *,
    group: str,
    value: str,
    window: int,
    aggregation: Literal["mean", "sum"] = "mean",
) -> pd.Series:
    """Calculate a rolling statistic from prior rows of the same entity only.

    ``groupby(...).shift().rolling()`` is subtly unsafe because ``rolling`` then
    operates on the flattened Series and can cross group boundaries.  Keeping
    the rolling operation inside ``transform`` preserves the original row index
    while guaranteeing that one club can never contribute to another club's
    history.
    """
    if window < 1:
        raise ValueError("window must be positive")
    if group not in frame or value not in frame:
        raise KeyError(f"Missing rolling input columns: {group!r}, {value!r}")

    def calculate(series: pd.Series) -> pd.Series:
        rolling = series.shift(1).rolling(window, min_periods=1)
        return rolling.mean() if aggregation == "mean" else rolling.sum()

    return frame.groupby(group, sort=False)[value].transform(calculate)


def chronological_split_indices(
    dates: Sequence[object], *, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Split observations by date, keeping equal-date fixtures in one partition."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    parsed = pd.to_datetime(pd.Series(dates), errors="coerce")
    if len(parsed) < 2:
        raise ValueError("At least two dated observations are required")
    if parsed.isna().any():
        raise ValueError("All observations require a valid date for temporal validation")

    ordered = np.argsort(parsed.to_numpy(), kind="stable")
    desired_test_rows = max(1, math.ceil(len(parsed) * test_size))
    boundary_position = len(parsed) - desired_test_rows
    cutoff = parsed.iloc[ordered[boundary_position]]
    train_indices = np.flatnonzero((parsed < cutoff).to_numpy())
    test_indices = np.flatnonzero((parsed >= cutoff).to_numpy())
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Dates do not provide distinct train and test periods")
    return train_indices, test_indices
