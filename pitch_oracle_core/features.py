"""Shared point-in-time feature policy for model training and inference."""

from __future__ import annotations

from collections.abc import Sequence
import math
import re
from typing import Literal

import numpy as np
import pandas as pd


def parse_match_dates(values: Sequence[object]) -> pd.Series:
    """Parse ISO dates and football-data's day-first dates without ambiguity."""
    source = pd.Series(values, copy=False)
    text = source.astype("string").str.strip()
    iso = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    result = pd.Series(pd.NaT, index=source.index, dtype="datetime64[ns]")
    result.loc[iso] = pd.to_datetime(text.loc[iso], format="%Y-%m-%d", errors="coerce")
    result.loc[~iso] = pd.to_datetime(text.loc[~iso], dayfirst=True, errors="coerce")
    return result


def completed_match_rows(
    frame: pd.DataFrame,
    *,
    date_column: str = "MatchDate",
    result_column: str = "FullTimeResult",
) -> pd.DataFrame:
    """Return dated, completed matches suitable for point-in-time features."""
    result = frame.copy()
    result[date_column] = parse_match_dates(result[date_column])
    valid = result[date_column].notna() & result[result_column].isin(("H", "D", "A"))
    return result.loc[valid].reset_index(drop=True)


FEATURE_POLICY_VERSION = 3

# football-data.co.uk mixes descriptive bookmaker fields with terse legacy
# codes (for example ``B365H``, ``PSCH`` and ``AvgCA``).  Looking only for the
# word "odds" therefore lets many market and closing-price fields leak into a
# supposedly football-only model.  Keep this policy centralized so training,
# auditing and serving cannot disagree about what "no odds" means.
_MARKET_NAME_TOKENS = (
    "odds", "impliedprob", "marketmargin", "oddsmovement", "bet365_value",
    "bet365_expectedtotalgoals", "bet365_homevs", "bet365_awayvs",
    "bet365_overunder_margin", "bet365_ah_margin",
)
_BOOKMAKER_NAME_PREFIXES = (
    "bet365_", "betwin_", "bluesquare_", "gamebookers_", "interwetten_",
    "ladbrokes_", "pinnacle_", "sportingbet_", "stanjames_", "stanleybet_",
    "vcbet_", "williamhill_", "betbrain_", "max_", "avg_",
)
_FOOTBALL_DATA_MARKET_CODE = re.compile(
    r"^(?:"
    r"B365|PS|BW|IW|WH|VC|LB|SB|SJ|SY|GB|BS|"
    r"Max|Avg|Bb|BF|BFE|BFD|BMGM|BV|CL|1XB|P"
    r")(?:C?(?:H|D|A)|C?[<>]2\.5|C?AH[HAh]|CAH[HAh]|.*Odds)$",
    re.IGNORECASE,
)
_EXACT_MARKET_CODES = {
    "AHh", "AHCh", "PAHH", "PAHA", "PCAHH", "PCAHA",
    "MaxAHH", "MaxAHA", "AvgAHH", "AvgAHA",
    "MaxCAHH", "MaxCAHA", "AvgCAHH", "AvgCAHA",
}

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


def is_market_feature(column: str) -> bool:
    """Return whether a field derives from bookmaker prices or market state."""
    name = str(column)
    lowered = name.lower()
    if any(token in lowered for token in _MARKET_NAME_TOKENS):
        return True
    if lowered.startswith(tuple(prefix.lower() for prefix in _BOOKMAKER_NAME_PREFIXES)):
        return True
    return name in _EXACT_MARKET_CODES or bool(_FOOTBALL_DATA_MARKET_CODE.fullmatch(name))


def no_odds_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return point-in-time feature columns that are independent of the market."""
    return [
        column for column in prematch_feature_columns(frame)
        if not is_market_feature(column)
    ]


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


def chronological_partition_indices(
    dates: Sequence[object], *, calibration_size: float = 0.2, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create train/calibration/test periods without splitting match dates."""
    if calibration_size <= 0 or test_size <= 0 or calibration_size + test_size >= 1:
        raise ValueError("calibration_size and test_size must be positive and sum to less than 1")
    train_calibration, test = chronological_split_indices(dates, test_size=test_size)
    remaining_dates = pd.to_datetime(pd.Series(dates), errors="coerce").iloc[train_calibration]
    relative_calibration = calibration_size / (1.0 - test_size)
    train_relative, calibration_relative = chronological_split_indices(
        remaining_dates, test_size=relative_calibration
    )
    return (
        train_calibration[train_relative],
        train_calibration[calibration_relative],
        test,
    )


def completed_future_rows(
    frame: pd.DataFrame,
    *,
    as_of: object | None = None,
    tolerance_days: int = 1,
    date_column: str = "MatchDate",
    result_column: str = "FullTimeResult",
) -> pd.DataFrame:
    """Return completed matches dated implausibly after the audit cutoff."""
    if tolerance_days < 0:
        raise ValueError("tolerance_days cannot be negative")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    cutoff = pd.Timestamp(as_of or pd.Timestamp.now(tz="UTC").date()).tz_localize(None)
    completed = frame[result_column].isin(("H", "D", "A"))
    return frame.loc[completed & (dates > cutoff + pd.Timedelta(days=tolerance_days))].copy()
