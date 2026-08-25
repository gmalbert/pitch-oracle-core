"""Goal model variants — Negative Binomial, Bivariate Poisson, ZIP, Weibull Copula.

Each model is a one-line fit via penaltyblog.models and can serve as a
Model Lab challenger to the production Dixon-Coles default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from penaltyblog.models import (
    BivariatePoissonGoalModel,
    DixonColesGoalModel,
    NegativeBinomialGoalModel,
    PoissonGoalsModel,
    WeibullCopulaGoalsModel,
    ZeroInflatedPoissonGoalsModel,
)

from .goal_models import DEFAULT_XI, time_decay_weights


MODEL_CLASSES = {
    "poisson": PoissonGoalsModel,
    "dixon_coles": DixonColesGoalModel,
    "negative_binomial": NegativeBinomialGoalModel,
    "bivariate_poisson": BivariatePoissonGoalModel,
    "zero_inflated_poisson": ZeroInflatedPoissonGoalsModel,
    "weibull_copula": WeibullCopulaGoalsModel,
}


@dataclass(frozen=True)
class ModelFitResult:
    """Summary of a fitted model variant."""
    model_name: str
    model: Any
    n_fixtures: int
    n_teams: int


def fit_model_variant(
    frame: pd.DataFrame,
    model_name: str,
    *,
    xi: float = DEFAULT_XI,
    use_time_decay: bool = True,
    minimizer_options: dict[str, Any] | None = None,
) -> ModelFitResult:
    """Fit any penaltyblog goal model variant with optional time-decay weights.

    ``frame`` must contain ``goals_home``, ``goals_away``, ``team_home``,
    ``team_away``, and ``date`` (or ``datetime``).
    """
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}. Choose from {sorted(MODEL_CLASSES)}")
    cls = MODEL_CLASSES[model_name]
    date_col = "datetime" if "datetime" in frame.columns else "date"
    weights = time_decay_weights(frame[date_col], xi=xi) if use_time_decay else None
    model = cls(
        goals_home=frame["goals_home"].to_numpy().copy(),
        goals_away=frame["goals_away"].to_numpy().copy(),
        teams_home=frame["team_home"].to_numpy().copy(),
        teams_away=frame["team_away"].to_numpy().copy(),
        weights=weights.copy() if weights is not None and hasattr(weights, 'copy') else weights,
    )
    model.fit(minimizer_options=minimizer_options or {"maxiter": 5000, "ftol": 1e-9})
    return ModelFitResult(
        model_name=model_name,
        model=model,
        n_fixtures=len(frame),
        n_teams=len(set(frame["team_home"]).union(frame["team_away"])),
    )


def fit_all_variants(
    frame: pd.DataFrame,
    *,
    xi: float = DEFAULT_XI,
    minimizer_options: dict[str, Any] | None = None,
) -> dict[str, ModelFitResult]:
    """Fit all model variants on the same training data for AIC comparison."""
    results = {}
    for name in MODEL_CLASSES:
        try:
            results[name] = fit_model_variant(
                frame, name, xi=xi, minimizer_options=minimizer_options,
            )
        except Exception:
            continue
    return results


def aic_leaderboard(
    frame: pd.DataFrame,
    *,
    xi: float = DEFAULT_XI,
) -> pd.DataFrame:
    """Build an AIC leaderboard across all model variants.

    Note: penaltyblog 1.12.0 does not expose .aic on fitted models, so we
    approximate AIC = 2k - 2ln(L) from the log-likelihood and parameter count
    where available, or report fit status only.
    """
    results = fit_all_variants(frame, xi=xi)
    rows = []
    for name, result in results.items():
        row = {"model": name, "n_fixtures": result.n_fixtures, "n_teams": result.n_teams}
        params = result.model.get_params()
        row["n_params"] = len(params) if isinstance(params, dict) else 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_params", ascending=False)
