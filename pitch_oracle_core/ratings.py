"""Rating systems backed by penaltyblog.ratings — Elo, Massey, Colley, Pi.

Provides a unified interface for computing and comparing team ratings
across four complementary systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from penaltyblog.ratings import Colley, Elo, Massey, PiRatingSystem


def _chronological_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Return a stable chronological frame when a date column is available."""
    for column in ("kickoff_utc", "datetime", "date", "match_date"):
        if column in matches.columns:
            frame = matches.copy()
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
            return frame.sort_values(column, kind="stable").reset_index(drop=True)
    return matches.reset_index(drop=True)


@dataclass(frozen=True)
class RatingSnapshot:
    """A point-in-time set of ratings from all four systems."""
    team: str
    elo: float | None = None
    massey: float | None = None
    colley: float | None = None
    pi: float | None = None
    elo_rank: int | None = None
    massey_rank: int | None = None
    colley_rank: int | None = None
    pi_rank: int | None = None


def compute_elo_ratings(
    matches: pd.DataFrame,
    *,
    k: float = 20.0,
    home_field_advantage: float = 100.0,
    home_col: str = "team_home",
    away_col: str = "team_away",
    goals_home_col: str = "goals_home",
    goals_away_col: str = "goals_away",
) -> tuple[Elo, list[dict[str, Any]]]:
    """Compute Elo ratings chronologically from match results.

    Returns the fitted Elo object and a list of per-match update records
    (for building Elo history time series).
    """
    matches = _chronological_matches(matches)
    elo = Elo(k=k, home_field_advantage=home_field_advantage)
    history: list[dict[str, Any]] = []
    for _, row in matches.iterrows():
        home, away = str(row[home_col]), str(row[away_col])
        gh, ga = int(row[goals_home_col]), int(row[goals_away_col])
        result = 0 if gh > ga else (2 if gh < ga else 1)
        pre_home = elo.ratings.get(home, 1500.0)
        pre_away = elo.ratings.get(away, 1500.0)
        elo.update_ratings(home, away, result)
        history.append({
            "team_home": home, "team_away": away,
            "goals_home": gh, "goals_away": ga,
            "elo_home_pre": pre_home, "elo_away_pre": pre_away,
            "elo_home_post": elo.ratings[home],
            "elo_away_post": elo.ratings[away],
            "known_at": row.get("kickoff_utc", row.get("datetime", row.get("date"))),
        })
    return elo, history


def compute_massey_ratings(
    matches: pd.DataFrame,
    *,
    home_col: str = "team_home",
    away_col: str = "team_away",
    goals_home_col: str = "goals_home",
    goals_away_col: str = "goals_away",
) -> pd.DataFrame:
    """Compute Massey ratings from a completed season."""
    massey = Massey(
        goals_home=matches[goals_home_col].to_numpy(),
        goals_away=matches[goals_away_col].to_numpy(),
        teams_home=matches[home_col].to_numpy(),
        teams_away=matches[away_col].to_numpy(),
    )
    return massey.get_ratings()


def compute_colley_ratings(
    matches: pd.DataFrame,
    *,
    include_draws: bool = True,
    home_col: str = "team_home",
    away_col: str = "team_away",
    goals_home_col: str = "goals_home",
    goals_away_col: str = "goals_away",
) -> pd.DataFrame:
    """Compute Colley ratings from a completed season."""
    colley = Colley(
        goals_home=matches[goals_home_col].to_numpy(),
        goals_away=matches[goals_away_col].to_numpy(),
        teams_home=matches[home_col].to_numpy(),
        teams_away=matches[away_col].to_numpy(),
        include_draws=include_draws,
    )
    return colley.get_ratings()


def compute_pi_ratings(
    matches: pd.DataFrame,
    *,
    alpha: float = 0.15,
    beta: float = 0.10,
    k: float = 0.5,
    sigma: float = 0.42,
    home_col: str = "team_home",
    away_col: str = "team_away",
    goals_home_col: str = "goals_home",
    goals_away_col: str = "goals_away",
) -> tuple[PiRatingSystem, list[dict[str, Any]]]:
    """Compute Pi ratings chronologically from match results."""
    pi = PiRatingSystem(alpha=alpha, beta=beta, k=k, sigma=sigma)
    history: list[dict[str, Any]] = []
    for _, row in matches.iterrows():
        home, away = str(row[home_col]), str(row[away_col])
        gh, ga = int(row[goals_home_col]), int(row[goals_away_col])
        gd = gh - ga
        pi.update_ratings(home, away, gd)
        probs = pi.calculate_match_probabilities(home, away)
        # Pi ratings store home/away sub-ratings as dicts
        home_rating = pi.team_ratings.get(home, {})
        away_rating = pi.team_ratings.get(away, {})
        home_pi = home_rating.get("home", 0.0) if isinstance(home_rating, dict) else float(home_rating)
        away_pi = away_rating.get("away", 0.0) if isinstance(away_rating, dict) else float(away_rating)
        history.append({
            "team_home": home, "team_away": away,
            "goals_home": gh, "goals_away": ga,
            "pi_home": home_pi,
            "pi_away": away_pi,
            "pi_p_home": probs.get("home_win", probs.get("home", 0.33)),
            "pi_p_draw": probs.get("draw", 0.33),
            "pi_p_away": probs.get("away_win", probs.get("away", 0.33)),
        })
    return pi, history


def build_combined_rankings(
    matches: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """Build a combined rankings table with all four rating systems side-by-side."""
    elo, _ = compute_elo_ratings(matches, **kwargs)
    massey_df = compute_massey_ratings(matches, **kwargs)
    colley_df = compute_colley_ratings(matches, **kwargs)
    pi, _ = compute_pi_ratings(matches, **kwargs)

    home_col = kwargs.get("home_col", "team_home")
    away_col = kwargs.get("away_col", "team_away")
    teams = sorted(
        set(matches[home_col].astype(str)).union(matches[away_col].astype(str))
    )
    rows = []
    for team in teams:
        massey_val = massey_df.loc[massey_df["team"] == team, "rating"]
        colley_val = colley_df.loc[colley_df["team"] == team, "rating"]
        # Pi ratings store home/away sub-ratings as dicts
        pi_rating = pi.team_ratings.get(team, {})
        pi_val = pi_rating.get("home", 0.0) if isinstance(pi_rating, dict) else float(pi_rating)
        rows.append({
            "team": team,
            "elo": elo.ratings.get(team, 1500.0),
            "massey": float(massey_val.iloc[0]) if len(massey_val) > 0 else 0.0,
            "colley": float(colley_val.iloc[0]) if len(colley_val) > 0 else 0.0,
            "pi": float(pi_val),
        })
    df = pd.DataFrame(rows)
    for col in ("elo", "massey", "colley", "pi"):
        df[f"{col}_rank"] = df[col].astype(float).rank(ascending=False).astype(int)
    return df.sort_values("elo_rank")


def elo_form_delta(
    elo_history: list[dict[str, Any]],
    team: str,
    window_days: int = 28,
) -> float | None:
    """Compute the Elo delta over a rolling window for one team.

    Returns the difference (current Elo - Elo N days ago), or None if
    insufficient history.
    """
    team_matches = [
        h for h in elo_history
        if h["team_home"] == team or h["team_away"] == team
    ]
    if len(team_matches) < 2:
        return None
    current = team_matches[-1]
    current_elo = current["elo_home_post"] if current["team_home"] == team else current["elo_away_post"]
    # Approximate window by match count (assumes ~2 matches per week)
    matches_per_week = 2
    window_matches = max(1, (window_days * matches_per_week) // 7)
    if len(team_matches) <= window_matches:
        return None
    past = team_matches[-(window_matches + 1)]
    past_elo = past["elo_home_post"] if past["team_home"] == team else past["elo_away_post"]
    return current_elo - past_elo
