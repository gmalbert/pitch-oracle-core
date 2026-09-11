"""FootballProbabilityGrid wrapper — 30+ markets from a single goal-model predict.

Wraps ``penaltyblog.models.FootballProbabilityGrid`` so every consumer
(Streamlit pages, best-bets, feature engineering) gets the full market surface
from one call without hand-rolling Poisson loops.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from penaltyblog.models import FootballProbabilityGrid, create_dixon_coles_grid
from pitch_oracle_core.domain.forecasts import markets_from_score_matrix
from pitch_oracle_core.domain.probability_grid import ProbabilityGrid


def _domain_grid(grid: FootballProbabilityGrid) -> ProbabilityGrid:
    """Convert an upstream grid without hiding represented tail mass."""
    mass = np.asarray(grid.grid, dtype=float)
    represented = float(mass.sum())
    if represented <= 0 or represented > 1.0 + 1e-8:
        raise ValueError(f"invalid probability-grid mass: {represented}")
    if represented > 1.0:
        mass = mass / represented
        represented = 1.0
    return ProbabilityGrid(
        mass=mass,
        tail_mass=max(0.0, 1.0 - represented),
        max_goals_home=mass.shape[0] - 1,
        max_goals_away=mass.shape[1] - 1,
    )


def _totals(matrix: np.ndarray, line: float) -> tuple[float, float, float]:
    if line < 0 or line % 1 not in (0.0, 0.5):
        raise ValueError("goal lines must be non-negative half-goal or integer values")
    home = np.arange(matrix.shape[0])[:, None]
    away = np.arange(matrix.shape[1])[None, :]
    total = home + away
    return (
        float(matrix[total < line].sum()),
        float(matrix[total == line].sum()) if line % 1 == 0 else 0.0,
        float(matrix[total > line].sum()),
    )


def market_row_from_domain(grid: ProbabilityGrid, fixture_id: str = "") -> dict[str, Any]:
    """Extract the canonical market surface from Pitch Oracle's grid contract."""
    matrix = grid.normalized_mass()
    markets = markets_from_score_matrix(matrix)
    under_25, _, over_25 = _totals(matrix, 2.5)
    under_35, _, over_35 = _totals(matrix, 3.5)
    home_mask = np.arange(matrix.shape[0])[:, None] > np.arange(matrix.shape[1])[None, :]
    away_mask = np.arange(matrix.shape[0])[:, None] <= np.arange(matrix.shape[1])[None, :]
    home_win = float(markets["p_home"])
    away_win = float(markets["p_away"])
    draw = float(markets["p_draw"])
    home_goals = np.arange(matrix.shape[0])[:, None]
    away_goals = np.arange(matrix.shape[1])[None, :]
    return {
        "fixture_id": fixture_id,
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "btts_yes": float(markets["p_btts_yes"]),
        "btts_no": float(markets["p_btts_no"]),
        "double_chance_1x": float(markets["p_home_or_draw"]),
        "double_chance_x2": float(markets["p_away_or_draw"]),
        "double_chance_12": float(markets["p_home_or_away"]),
        "draw_no_bet_home": home_win / (home_win + away_win),
        "draw_no_bet_away": away_win / (home_win + away_win),
        "win_to_nil_home": float(markets["p_home_win_to_nil"]),
        "win_to_nil_away": float(markets["p_away_win_to_nil"]),
        "over_2_5": over_25,
        "under_2_5": under_25,
        "over_3_5": over_35,
        "under_3_5": under_35,
        "exp_home_goals": float((matrix * home_goals).sum()),
        "exp_away_goals": float((matrix * away_goals).sum()),
        "exp_points_home": 3.0 * home_win + draw,
        "exp_points_away": 3.0 * away_win + draw,
        "ah_home_minus0_5_win": float(matrix[home_mask].sum()),
        "ah_home_minus0_5_push": 0.0,
        "ah_home_minus0_5_lose": float(1.0 - matrix[home_mask].sum()),
        "ah_away_plus0_5_win": float(matrix[away_mask].sum()),
        "ah_away_plus0_5_push": 0.0,
        "ah_away_plus0_5_lose": float(1.0 - matrix[away_mask].sum()),
        "score_grid_tail_mass": float(grid.tail_mass),
    }


def market_row(grid: FootballProbabilityGrid, fixture_id: str = "") -> dict[str, Any]:
    """Extract every market from a grid into a flat dict suitable for a DataFrame row."""
    return market_row_from_domain(_domain_grid(grid), fixture_id=fixture_id)


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
