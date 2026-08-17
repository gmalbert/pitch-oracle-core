"""League trends, competitive balance, and aligned cross-league exports."""

from __future__ import annotations

import numpy as np
import pandas as pd


def league_trends(matches: pd.DataFrame, *, window: int = 50) -> pd.DataFrame:
    frame = matches.sort_values("kickoff_utc").copy()
    frame["total_goals"] = frame.home_goals + frame.away_goals
    frame["home_win"] = (frame.home_goals > frame.away_goals).astype(float)
    frame["draw"] = (frame.home_goals == frame.away_goals).astype(float)
    frame["home_advantage_goals"] = frame.home_goals - frame.away_goals
    for column in ("total_goals", "home_win", "draw", "home_advantage_goals"):
        frame[f"rolling_{column}"] = frame[column].rolling(window, min_periods=10).mean()
    optional_sources = {
        "cards": ("cards",),
        "tempo": ("tempo", "shots", "total_shots"),
        "market_error": ("market_error", "market_log_loss_error"),
    }
    for target, candidates in optional_sources.items():
        source = next((column for column in candidates if column in frame), None)
        if source is not None:
            frame[f"rolling_{target}"] = pd.to_numeric(
                frame[source], errors="coerce"
            ).rolling(window, min_periods=10).mean()
    frame["sample_n"] = frame["fixture_id"].rolling(window, min_periods=1).count()
    frame["window_matches"] = int(window)
    return frame


def competitive_balance(
    ratings: pd.Series, title_probabilities: pd.Series
) -> dict[str, float]:
    values = ratings.astype(float).to_numpy()
    probabilities = title_probabilities.astype(float).to_numpy()
    probabilities = probabilities / probabilities.sum()
    n = len(values)
    return {
        "normalized_strength_dispersion": float(np.std(values) / max(abs(np.mean(values)), 1e-9)),
        "title_herfindahl": float(np.square(probabilities).sum()),
        "parity_index": float(1 - np.square(probabilities).sum()) / max(1 - 1 / n, 1e-9),
    }


def cross_league_export(
    *,
    competition_id: str,
    edition_id: str,
    matches: pd.DataFrame,
    calibration_error: float,
    rating_dispersion: float,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "competition_id": competition_id,
        "edition_id": edition_id,
        "fixtures": len(matches),
        "goals_per_match": float((matches.home_goals + matches.away_goals).mean()),
        "home_advantage_goals": float((matches.home_goals - matches.away_goals).mean()),
        "draw_rate": float((matches.home_goals == matches.away_goals).mean()),
        "calibration_error": float(calibration_error),
        "rating_dispersion": float(rating_dispersion),
    }])
