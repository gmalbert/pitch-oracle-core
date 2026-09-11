"""Bayesian goal models — credible intervals and uncertainty features.

Wraps ``penaltyblog.models.BayesianGoalModel`` and
``HierarchicalBayesianGoalModel`` for uncertainty-aware pricing and
feature engineering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from penaltyblog.models import (
    BayesianGoalModel,
    HierarchicalBayesianGoalModel,
    create_dixon_coles_grid,
)

from .goal_models import DEFAULT_XI, time_decay_weights


@dataclass(frozen=True)
class BayesianFitResult:
    """Result of fitting a Bayesian goal model."""
    model: Any
    n_fixtures: int
    n_teams: int
    diagnostics: dict[str, Any]


def fit_bayesian(
    frame: pd.DataFrame,
    *,
    n_samples: int = 2000,
    burn: int = 1000,
    n_chains: int = 4,
    xi: float = DEFAULT_XI,
    hierarchical: bool = False,
) -> BayesianFitResult:
    """Fit a Bayesian goal model with time-decay weights.

    If ``hierarchical=True``, uses ``HierarchicalBayesianGoalModel`` which
    shares information across teams — best for sparse leagues.
    """
    date_col = "datetime" if "datetime" in frame.columns else "date"
    weights = time_decay_weights(frame[date_col], xi=xi)
    cls = HierarchicalBayesianGoalModel if hierarchical else BayesianGoalModel
    model = cls(
        goals_home=frame["goals_home"].to_numpy().copy(),
        goals_away=frame["goals_away"].to_numpy().copy(),
        teams_home=frame["team_home"].to_numpy().copy(),
        teams_away=frame["team_away"].to_numpy().copy(),
        weights=weights.copy() if hasattr(weights, 'copy') else weights,
    )
    model.fit(n_samples=n_samples, burn=burn, n_chains=n_chains)
    diagnostics = model.get_diagnostics() if hasattr(model, 'get_diagnostics') else {}
    return BayesianFitResult(
        model=model,
        n_fixtures=len(frame),
        n_teams=len(set(frame["team_home"]).union(frame["team_away"])),
        diagnostics=diagnostics,
    )


def bayesian_predict_grid(
    model: Any,
    home_team: str,
    away_team: str,
    max_goals: int = 15,
) -> dict[str, float]:
    """Predict with a Bayesian model, returning the posterior-predictive grid."""
    grid = model.predict(home_team, away_team, max_goals=max_goals)
    return {
        "home_win": grid.home_win,
        "draw": grid.draw,
        "away_win": grid.away_win,
        "btts_yes": grid.btts_yes,
        "over_2_5": grid.totals(2.5)[2],
    }


def posterior_probability_draws(
    model: Any,
    home_team: str,
    away_team: str,
    *,
    max_goals: int = 15,
    n_samples: int = 100,
    neutral_venue: bool = False,
) -> np.ndarray:
    """Return posterior draws of ``(home, draw, away)`` probabilities.

    ``BayesianGoalModel`` stores the posterior trace and exposes the team index
    map. The calculation mirrors its Dixon-Coles parameterization while using
    the public grid constructor for market aggregation. This function is kept
    in a narrow adapter so a future upstream parameterization change is caught
    by one contract test.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    trace = np.asarray(getattr(model, "trace", None), dtype=float)
    team_to_idx = getattr(model, "team_to_idx", None)
    n_teams = int(getattr(model, "n_teams", 0))
    if trace.ndim != 2 or team_to_idx is None or n_teams <= 0:
        raise ValueError("fitted Bayesian model does not expose a posterior trace")
    if home_team not in team_to_idx or away_team not in team_to_idx:
        raise KeyError(f"Unknown team: {home_team!r} or {away_team!r}")
    home_index = int(team_to_idx[home_team])
    away_index = int(team_to_idx[away_team])
    sample_indices = np.linspace(0, len(trace) - 1, min(n_samples, len(trace)), dtype=int)
    rows = []
    for index in sample_indices:
        params = trace[index]
        home_attack = params[home_index]
        away_attack = params[away_index]
        home_defense = params[n_teams + home_index]
        away_defense = params[n_teams + away_index]
        home_advantage = 0.0 if neutral_venue else params[-2]
        rho = params[-1]
        home_lambda = float(np.exp(home_attack + away_defense + home_advantage))
        away_lambda = float(np.exp(away_attack + home_defense))
        grid = create_dixon_coles_grid(
            home_lambda, away_lambda, rho=float(rho), max_goals=max_goals
        )
        rows.append([grid.home_win, grid.draw, grid.away_win])
    return np.asarray(rows, dtype=float)


def uncertainty_features(
    model: Any,
    fixtures: pd.DataFrame,
    home_col: str = "team_home",
    away_col: str = "team_away",
    n_samples: int = 100,
) -> pd.DataFrame:
    """Compute uncertainty features from posterior samples for ML ensemble.

    Returns per-fixture features: p_home_mean, p_home_std, p_home_ci_width,
    p_draw_mean, p_away_mean.
    """
    rows = []
    for _, row in fixtures.iterrows():
        home, away = str(row[home_col]), str(row[away_col])
        try:
            grid = model.predict(home, away, max_goals=10)
            draws = posterior_probability_draws(
                model, home, away, max_goals=10, n_samples=n_samples
            )
            lower, median, upper = np.quantile(draws, [0.05, 0.50, 0.95], axis=0)
            rows.append({
                "team_home": home,
                "team_away": away,
                "bayes_p_home_mean": grid.home_win,
                "bayes_p_draw_mean": grid.draw,
                "bayes_p_away_mean": grid.away_win,
                "bayes_p_home_std": float(draws[:, 0].std(ddof=0)),
                "bayes_p_home_p05": float(lower[0]),
                "bayes_p_home_p50": float(median[0]),
                "bayes_p_home_p95": float(upper[0]),
                "bayes_p_home_ci_width": float(upper[0] - lower[0]),
            })
        except (KeyError, ValueError):
            continue
    return pd.DataFrame(rows)
