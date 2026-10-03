"""Shot shape summaries with explicit distinction between absent and zero shots."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitch_oracle_core.pitchapi.normalize import boolean

SHOT_METRICS = ("xg", "xgot", "shots", "shots_on_target", "xg_per_shot", "xgot_per_sot", "xgot_minus_xg", "inside_box_share", "set_piece_xg_share", "open_play_xg_share", "blocked_rate", "header_share", "chance_concentration", "finishing_vs_expectation")


def _numeric(frame: pd.DataFrame, field: str) -> pd.Series:
    return pd.to_numeric(frame.get(field, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _flags(frame: pd.DataFrame, field: str) -> pd.Series:
    return frame.get(field, pd.Series(None, index=frame.index, dtype=object)).map(boolean)


def _ratio(value: float, denominator: float) -> float:
    return value / denominator if np.isfinite(denominator) and denominator > 0 else np.nan


def _complete_sum(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(values.sum()) if values.notna().all() else np.nan


def chance_concentration(xg: pd.Series, top_n: int = 3) -> float:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    values = pd.to_numeric(xg, errors="coerce")
    if (values.dropna() < 0).any():
        raise ValueError("Expected goals cannot be negative")
    if values.empty or values.isna().any() or values.sum() <= 0:
        return np.nan
    return float(values.nlargest(top_n).sum() / values.sum())


def shot_profile_for_team(shots: pd.DataFrame, *, available: bool = True) -> dict[str, float]:
    if not available:
        return {name: np.nan for name in SHOT_METRICS}
    # Shootouts are not part of regular-time match xG or forecasting targets.
    periods = shots.get("period", pd.Series("", index=shots.index)).astype(str).str.casefold()
    own_goals = _flags(shots, "is_own_goal").eq(True)
    shots = shots.loc[~periods.str.contains("shootout|penaltyshoot", regex=True) & ~own_goals].copy()
    xg, xgot = _numeric(shots, "expected_goals"), _numeric(shots, "expected_goals_on_target")
    if (xg.dropna() < 0).any() or (xgot.dropna() < 0).any():
        raise ValueError("Expected goals cannot be negative")
    n = len(shots)
    on_target = _flags(shots, "is_on_target")
    # Non-on-target shots have no post-shot chance by definition; unknown on-target xGOT stays unknown.
    xgot = xgot.mask(on_target.eq(False), 0.0)
    total_xg, total_xgot = _complete_sum(xg), _complete_sum(xgot)
    sot = float(on_target.eq(True).sum()) if on_target.notna().all() else np.nan
    situation = shots.get("situation", pd.Series(None, index=shots.index, dtype=object))
    descriptions = situation.fillna("").astype(str).str.casefold()
    set_piece = descriptions.str.contains("corner|free.?kick|set.?piece|penalty", regex=True)
    open_play = descriptions.str.contains("regular|open.?play|fast.?break", regex=True)
    known_situations = situation.notna().all() and (set_piece | open_play).all()
    body = shots.get("body_part", shots.get("shot_type", pd.Series(None, index=shots.index, dtype=object)))
    result = {
        "xg": total_xg, "xgot": total_xgot, "shots": float(n), "shots_on_target": sot,
        "xg_per_shot": _ratio(total_xg, n), "xgot_per_sot": _ratio(total_xgot, sot),
        "xgot_minus_xg": total_xgot - total_xg,
        "set_piece_xg_share": _ratio(float(xg.loc[set_piece].sum()), total_xg) if known_situations else np.nan,
        "open_play_xg_share": _ratio(float(xg.loc[open_play].sum()), total_xg) if known_situations else np.nan,
        "header_share": _ratio(float(body.fillna("").astype(str).str.casefold().str.contains("head").sum()), n) if body.notna().all() else np.nan,
        "chance_concentration": chance_concentration(xg),
        "finishing_vs_expectation": float(_flags(shots, "is_goal").eq(True).sum()) - total_xg if _flags(shots, "is_goal").notna().all() else np.nan,
    }
    for source, target in (("is_inside_box", "inside_box_share"), ("is_blocked", "blocked_rate")):
        flags = _flags(shots, source)
        result[target] = _ratio(float(flags.eq(True).sum()), n) if flags.notna().all() else np.nan
    return result


def build_match_shot_features(shots: pd.DataFrame, fixture_teams: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fixture in fixture_teams.itertuples(index=False):
        selected = shots.loc[shots.fixture_id.astype(str) == str(fixture.fixture_id)]
        available = bool(getattr(fixture, "shots_available", not selected.empty))
        row = {"fixture_id": fixture.fixture_id}
        if not selected.empty and "observed_at" in selected:
            row["observed_at"] = pd.to_datetime(selected.observed_at, utc=True).max().isoformat()
        elif hasattr(fixture, "shots_observed_at"):
            row["observed_at"] = fixture.shots_observed_at
        for side in ("home", "away"):
            team_id = getattr(fixture, f"{side}_team_id")
            profile = shot_profile_for_team(selected.loc[selected.team_id == team_id], available=available)
            row.update({f"{side}_{key}": value for key, value in profile.items()})
        rows.append(row)
    return pd.DataFrame(rows)
