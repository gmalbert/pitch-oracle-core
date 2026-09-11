"""Adapters from penaltyblog goal models to Pitch Oracle's score protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from penaltyblog.models import DixonColesGoalModel

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid

from pitch_oracle_core.goal_models import DEFAULT_XI, time_decay_weights
from .protocol import FixtureFeatures, ForecastTrack, ModelSpec


REQUIRED_COLUMNS = frozenset(
    {"team_home", "team_away", "goals_home", "goals_away"}
)


def _default_spec() -> ModelSpec:
    return ModelSpec(
        model_id="pb-dixon-coles:v1",
        family="dixon_coles",
        track=ForecastTrack.INDEPENDENT,
        required_capabilities=frozenset({"historical_goals", "team_registry"}),
        hyperparameters={"xi": DEFAULT_XI, "max_goals": 15, "use_gradient": True},
    )


@dataclass
class PenaltyBlogDixonColes:
    """Implement ``ScoreModel`` using penaltyblog's Dixon-Coles model.

    The adapter intentionally requests an unnormalised upstream grid and
    converts truncation into the domain ``ProbabilityGrid.tail_mass`` field.
    This keeps tail loss visible to artifact validation and model promotion.
    """

    xi: float = DEFAULT_XI
    minimizer_options: dict[str, Any] = field(
        default_factory=lambda: {"maxiter": 5000, "ftol": 1e-9}
    )
    max_goals: int = 15
    spec: ModelSpec = field(default_factory=_default_spec)
    model: DixonColesGoalModel | None = field(default=None, init=False)
    fitted_at: datetime | None = field(default=None, init=False)

    def fit(self, matches: pd.DataFrame, *, cutoff_utc: datetime) -> "PenaltyBlogDixonColes":
        missing = REQUIRED_COLUMNS.difference(matches.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        date_col = "datetime" if "datetime" in matches.columns else "date"
        if date_col not in matches.columns:
            raise ValueError("matches require a datetime or date column")
        frame = matches.copy()
        frame[date_col] = pd.to_datetime(frame[date_col], utc=True, errors="raise")
        weights = frame.get("recency_weight")
        if weights is None:
            weights = time_decay_weights(frame[date_col], xi=self.xi, base_date=cutoff_utc)
        neutral = frame.get("neutral_venue")
        self.model = DixonColesGoalModel(
            goals_home=frame["goals_home"].to_numpy(copy=True),
            goals_away=frame["goals_away"].to_numpy(copy=True),
            teams_home=frame["team_home"].astype(str).to_numpy(copy=True),
            teams_away=frame["team_away"].astype(str).to_numpy(copy=True),
            weights=np.asarray(weights, dtype=float).copy(),
            neutral_venue=None if neutral is None else np.asarray(neutral, dtype=bool),
        )
        self.model.fit(minimizer_options=dict(self.minimizer_options), use_gradient=True)
        self.fitted_at = datetime.now().astimezone()
        return self

    def predict_grid(self, fixture: FixtureFeatures) -> ProbabilityGrid:
        if self.model is None:
            raise RuntimeError("model must be fitted before prediction")
        upstream = self.model.predict(
            fixture.home_team_id,
            fixture.away_team_id,
            max_goals=self.max_goals,
            normalize=False,
            neutral_venue=fixture.neutral_venue,
        )
        mass = np.asarray(upstream.grid, dtype=float)
        represented = float(mass.sum())
        if represented <= 0 or represented > 1.0 + 1e-8:
            raise ValueError(f"penaltyblog returned invalid represented mass: {represented}")
        if represented > 1.0:
            mass = mass / represented
            represented = 1.0
        return ProbabilityGrid(
            mass=mass,
            tail_mass=max(0.0, 1.0 - represented),
            max_goals_home=mass.shape[0] - 1,
            max_goals_away=mass.shape[1] - 1,
        )
