"""Rank-first Poisson challenger with point-in-time fixture covariates.

The model intentionally remains a goals model: match-outcome and totals markets are
derived from its joint score grid instead of being fitted by unrelated classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid
from .independent_poisson import independent_poisson_grid
from .protocol import FixtureFeatures, ForecastTrack, ModelSpec


DEFAULT_FEATURES = (
    "rank_difference",
    "elo_difference",
    "rest_difference",
    "form_difference",
)


@dataclass
class RankCovariateGoalsModel:
    """Two regularized Poisson heads sharing one pre-match feature contract."""

    feature_names: tuple[str, ...] = DEFAULT_FEATURES
    alpha: float = 1.0
    home_estimator: Pipeline | None = field(default=None, init=False, repr=False)
    away_estimator: Pipeline | None = field(default=None, init=False, repr=False)
    spec: ModelSpec = field(init=False)

    def __post_init__(self) -> None:
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and unique")
        if self.alpha < 0:
            raise ValueError("alpha cannot be negative")
        self.spec = ModelSpec(
            model_id="rank-covariate-poisson:v1",
            family="rank_covariate_poisson",
            track=ForecastTrack.INDEPENDENT,
            required_capabilities=frozenset(),
            hyperparameters={"alpha": self.alpha, "features": self.feature_names},
        )

    @staticmethod
    def _derive_features(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        pairs = {
            "rank_difference": ("home_rank", "away_rank"),
            "elo_difference": ("home_elo", "away_elo"),
            "rest_difference": ("home_rest_days", "away_rest_days"),
            "form_difference": ("home_form_points", "away_form_points"),
        }
        for target, (home, away) in pairs.items():
            if target not in result and {home, away}.issubset(result.columns):
                result[target] = result[home].astype(float) - result[away].astype(float)
        return result

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        derived = self._derive_features(frame)
        missing = set(self.feature_names).difference(derived.columns)
        if missing:
            raise ValueError(f"Missing rank/covariate features: {sorted(missing)}")
        design = derived.loc[:, self.feature_names].astype(float)
        if design.isna().any().any() or not np.isfinite(design.to_numpy()).all():
            raise ValueError("rank/covariate features must be finite")
        return design.to_numpy()

    def fit(
        self, matches: pd.DataFrame, *, cutoff_utc: datetime
    ) -> "RankCovariateGoalsModel":
        if cutoff_utc.tzinfo is None:
            raise ValueError("cutoff_utc must be timezone-aware")
        required = {"kickoff_utc", "home_goals", "away_goals"}
        missing = required.difference(matches.columns)
        if missing:
            raise ValueError(f"Missing training columns: {sorted(missing)}")
        frame = matches.copy()
        frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True)
        cutoff = pd.Timestamp(cutoff_utc)
        frame = frame.loc[
            (frame["kickoff_utc"] < cutoff)
            & frame["home_goals"].notna()
            & frame["away_goals"].notna()
        ]
        if len(frame) < max(4, len(self.feature_names) + 1):
            raise ValueError("insufficient pre-cutoff matches for rank/covariate fit")
        design = self._design(frame)

        def estimator() -> Pipeline:
            return Pipeline([
                ("scale", StandardScaler()),
                ("poisson", PoissonRegressor(alpha=self.alpha, max_iter=2_000)),
            ])

        self.home_estimator = estimator().fit(design, frame["home_goals"].astype(float))
        self.away_estimator = estimator().fit(design, frame["away_goals"].astype(float))
        return self

    def predict_rates(self, fixture: FixtureFeatures) -> tuple[float, float]:
        if self.home_estimator is None or self.away_estimator is None:
            raise RuntimeError("RankCovariateGoalsModel is not fitted")
        design = self._design(pd.DataFrame([fixture.values]))
        home = float(self.home_estimator.predict(design)[0])
        away = float(self.away_estimator.predict(design)[0])
        return max(home, 1e-6), max(away, 1e-6)

    def predict_grid(self, fixture: FixtureFeatures) -> ProbabilityGrid:
        return independent_poisson_grid(*self.predict_rates(fixture))
