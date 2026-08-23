"""FootballProbabilityGrid wrapper — 30+ markets from a single goal-model predict.

Wraps ``penaltyblog.models.FootballProbabilityGrid`` so every consumer
(Streamlit pages, best-bets, feature engineering) gets the full market surface
from one call without hand-rolling Poisson loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from penaltyblog.models import FootballProbabilityGrid, create_dixon_coles_grid


def market_row(grid: FootballProbabilityGrid, fixture_id: str = "") -> dict[str, Any]:
    """Extract every market from a grid into a flat dict suitable for a DataFrame row."""
    under_2, push_2, over_2 = grid.totals(2.5)
    under_3, push_3, over_3 = grid.totals(3.5)
    ah_home = grid.asian_handicap_probs("home", -0.5)
    ah_away = grid.asian_handicap_probs("away", 0.5)
    return {
        "fixture_id": fixture_id,
        "home_win": grid.home_win,
        "draw": grid.draw,
        "away_win": grid.away_win,
        "btts_yes": grid.btts_yes,
        "btts_no": grid.btts_no,
        "double_chance_1x": grid.double_chance_1x,
        "double_chance_x2": grid.double_chance_x2,
        "double_chance_12": grid.double_chance_12,
        "draw_no_bet_home": grid.draw_no_bet_home,
        "draw_no_bet_away": grid.draw_no_bet_away,
        "win_to_nil_home": grid.win_to_nil_home,
        "win_to_nil_away": grid.win_to_nil_away,
        "over_2_5": over_2,
        "under_2_5": under_2,
        "over_3_5": over_3,
        "under_3_5": under_3,
        "exp_home_goals": grid.home_goal_expectation,
        "exp_away_goals": grid.away_goal_expectation,
        "exp_points_home": grid.expected_points_home(),
        "exp_points_away": grid.expected_points_away(),
        "ah_home_minus0_5_win": ah_home["win"],
        "ah_home_minus0_5_push": ah_home["push"],
        "ah_home_minus0_5_lose": ah_home["lose"],
        "ah_away_plus0_5_win": ah_away["win"],
        "ah_away_plus0_5_push": ah_away["push"],
        "ah_away_plus0_5_lose": ah_away["lose"],
    }


def market_grid(
    model: Any,
    home_team: str,
    away_team: str,
    max_goals: int = 15,
    fixture_id: str = "",
) -> dict[str, Any]:
    """Predict and extract the full market surface for one fixture."""
    grid = model.predict(home_team, away_team, max_goals=max_goals)
    return market_row(grid, fixture_id=fixture_id)


def market_grid_from_lambdas(
    home_lambda: float,
    away_lambda: float,
    rho: float = -0.05,
    max_goals: int = 15,
    fixture_id: str = "",
) -> dict[str, Any]:
    """Build a market surface from external goal expectancies (e.g. XGBoost output)."""
    grid = create_dixon_coles_grid(home_lambda, away_lambda, rho=rho, max_goals=max_goals)
    return market_row(grid, fixture_id=fixture_id)


def slate_markets(
    model: Any,
    fixtures: pd.DataFrame,
    home_col: str = "team_home",
    away_col: str = "team_away",
    max_goals: int = 15,
) -> pd.DataFrame:
    """Vectorised market extraction for an entire fixture slate."""
    rows = []
    for idx, row in fixtures.iterrows():
        fixture_id = str(idx) if fixtures.index.name else ""
        try:
            rows.append(market_grid(model, row[home_col], row[away_col],
                                    max_goals=max_goals, fixture_id=fixture_id))
        except (KeyError, ValueError):
            continue
    return pd.DataFrame(rows)
