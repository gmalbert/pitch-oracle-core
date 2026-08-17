"""Time-decayed Dixon-Coles goals model."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid
from .independent_poisson import independent_poisson_grid
from .protocol import FixtureFeatures, ForecastTrack, ModelSpec


def poisson_log_pmf(goals: int, rate: float) -> float:
    if goals < 0 or rate <= 0 or not np.isfinite(rate):
        raise ValueError("invalid Poisson goal/rate")
    return goals * log(rate) - rate - lgamma(goals + 1.0)


def dc_tau(
    home_goals: int, away_goals: int, home_rate: float, away_rate: float, rho: float
) -> float:
    value = 1.0
    if home_goals == 0 and away_goals == 0:
        value = 1.0 - home_rate * away_rate * rho
    elif home_goals == 0 and away_goals == 1:
        value = 1.0 + home_rate * rho
    elif home_goals == 1 and away_goals == 0:
        value = 1.0 + away_rate * rho
    elif home_goals == 1 and away_goals == 1:
        value = 1.0 - rho
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Dixon-Coles parameters generated non-positive mass")
    return value


@dataclass(frozen=True)
class DixonColesFit:
    team_ids: tuple[str, ...]
    attack: np.ndarray
    defense: np.ndarray
    intercept: float
    home_advantage: float
    rho: float
    fitted_at: pd.Timestamp
    half_life_days: float


class DixonColesModel:
    def __init__(self, half_life_days: float = 365.0, l2: float = 0.02) -> None:
        if half_life_days <= 0 or l2 < 0:
            raise ValueError("half-life must be positive and l2 non-negative")
        self.half_life_days = half_life_days
        self.l2 = l2
        self.fit_: DixonColesFit | None = None

    def fit(self, matches: pd.DataFrame, as_of: object | None = None) -> "DixonColesModel":
        required = {
            "kickoff_utc", "home_team_id", "away_team_id", "home_goals", "away_goals"
        }
        missing = required.difference(matches.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        frame = matches.dropna(subset=list(required)).copy()
        frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True)
        cutoff = pd.Timestamp(as_of or frame["kickoff_utc"].max())
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        frame = frame.loc[frame["kickoff_utc"] < cutoff]
        teams = tuple(sorted(set(frame.home_team_id) | set(frame.away_team_id)))
        if len(teams) < 2 or len(frame) < len(teams):
            raise ValueError("Insufficient history for Dixon-Coles fit")
        index = {team_id: position for position, team_id in enumerate(teams)}
        home_index = frame.home_team_id.map(index).to_numpy()
        away_index = frame.away_team_id.map(index).to_numpy()
        home_goals = frame.home_goals.astype(int).to_numpy()
        away_goals = frame.away_goals.astype(int).to_numpy()
        if (home_goals < 0).any() or (away_goals < 0).any():
            raise ValueError("Goals cannot be negative")
        age_days = (cutoff - frame.kickoff_utc).dt.total_seconds().to_numpy() / 86_400
        weights = np.exp(-np.log(2.0) * age_days / self.half_life_days)
        size = len(teams)

        def unpack(parameters: np.ndarray):
            attack = parameters[:size]
            defense = parameters[size : 2 * size]
            attack = attack - attack.mean()
            defense = defense - defense.mean()
            intercept, home_advantage, rho = parameters[-3:]
            return attack, defense, intercept, home_advantage, rho

        def objective(parameters: np.ndarray) -> float:
            attack, defense, intercept, home_advantage, rho = unpack(parameters)
            home_rate = np.exp(
                intercept + home_advantage + attack[home_index] - defense[away_index]
            )
            away_rate = np.exp(intercept + attack[away_index] - defense[home_index])
            log_likelihood = np.empty(len(frame), dtype=float)
            for row in range(len(frame)):
                try:
                    tau = dc_tau(
                        home_goals[row], away_goals[row], home_rate[row], away_rate[row], rho
                    )
                except ValueError:
                    return 1e12
                log_likelihood[row] = (
                    log(tau)
                    + poisson_log_pmf(home_goals[row], home_rate[row])
                    + poisson_log_pmf(away_goals[row], away_rate[row])
                )
            penalty = self.l2 * (np.square(attack).sum() + np.square(defense).sum())
            return float(-(weights * log_likelihood).sum() + penalty)

        initial = np.zeros(2 * size + 3, dtype=float)
        initial[-3] = log(
            max((home_goals.sum() + away_goals.sum()) / (2 * len(frame)), 0.2)
        )
        initial[-2] = 0.15
        bounds = [(None, None)] * (2 * size + 2) + [(-0.2, 0.2)]
        result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Dixon-Coles fit failed: {result.message}")
        attack, defense, intercept, home_advantage, rho = unpack(result.x)
        self.fit_ = DixonColesFit(
            teams,
            attack,
            defense,
            float(intercept),
            float(home_advantage),
            float(rho),
            cutoff,
            self.half_life_days,
        )
        return self

    def expected_goals(self, home_team_id: str, away_team_id: str) -> tuple[float, float]:
        if self.fit_ is None:
            raise RuntimeError("Model is not fitted")
        index = {team_id: position for position, team_id in enumerate(self.fit_.team_ids)}
        if home_team_id not in index or away_team_id not in index:
            raise KeyError("Unknown team; resolve an explicit promoted-team prior first")
        home, away = index[home_team_id], index[away_team_id]
        home_rate = np.exp(
            self.fit_.intercept
            + self.fit_.home_advantage
            + self.fit_.attack[home]
            - self.fit_.defense[away]
        )
        away_rate = np.exp(
            self.fit_.intercept + self.fit_.attack[away] - self.fit_.defense[home]
        )
        return float(home_rate), float(away_rate)

    def score_grid(
        self, home_team_id: str, away_team_id: str, max_goals: int = 12
    ) -> ProbabilityGrid:
        if self.fit_ is None:
            raise RuntimeError("Model is not fitted")
        home_rate, away_rate = self.expected_goals(home_team_id, away_team_id)
        matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                probability = np.exp(
                    poisson_log_pmf(home_goals, home_rate)
                    + poisson_log_pmf(away_goals, away_rate)
                )
                tau = dc_tau(home_goals, away_goals, home_rate, away_rate, self.fit_.rho)
                if tau < 0:
                    raise ValueError("Dixon-Coles parameters generated negative mass")
                matrix[home_goals, away_goals] = probability * tau
        if matrix.sum() <= 0:
            raise ValueError("Invalid Dixon-Coles score distribution")
        represented = float(matrix.sum())
        if represented > 1.0 and represented - 1.0 < 1e-10:
            matrix /= represented
            represented = 1.0
        tail = max(0.0, 1.0 - represented)
        # The DC correction preserves total mass on infinite support. Truncation can
        # leave tiny numerical drift; retain it explicitly rather than hiding it.
        if represented > 1.0:
            matrix /= represented
            tail = 0.0
        return ProbabilityGrid(matrix, tail, max_goals, max_goals)

    def score_matrix(
        self, home_team_id: str, away_team_id: str, max_goals: int = 10
    ) -> np.ndarray:
        grid = self.score_grid(home_team_id, away_team_id, max_goals=max_goals)
        return grid.mass / grid.mass.sum()


class DixonColesForecaster:
    """Protocol adapter with an explicit independent-Poisson failure fallback."""

    def __init__(self, half_life_days: float = 365.0, l2: float = 0.02) -> None:
        self.model = DixonColesModel(half_life_days=half_life_days, l2=l2)
        self.spec = ModelSpec(
            model_id="dixon-coles-with-fallback:v1",
            family="dixon_coles",
            track=ForecastTrack.INDEPENDENT,
            required_capabilities=frozenset(),
            hyperparameters={"half_life_days": half_life_days, "l2": l2},
        )
        self.fallback_reason: str | None = None
        self.fallback_home_rate = 1.35
        self.fallback_away_rate = 1.10
        self.predictions = 0
        self.fallback_predictions = 0

    def fit(
        self, matches: pd.DataFrame, *, cutoff_utc
    ) -> "DixonColesForecaster":
        frame = matches.copy()
        if {"kickoff_utc", "home_goals", "away_goals"}.issubset(frame.columns):
            kickoff = pd.to_datetime(frame["kickoff_utc"], utc=True)
            usable = frame.loc[
                (kickoff < pd.Timestamp(cutoff_utc))
                & frame.home_goals.notna() & frame.away_goals.notna()
            ]
            if not usable.empty:
                self.fallback_home_rate = max(float(usable.home_goals.mean()), 0.1)
                self.fallback_away_rate = max(float(usable.away_goals.mean()), 0.1)
        try:
            self.model.fit(frame, as_of=cutoff_utc)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
        else:
            self.fallback_reason = None
        return self

    def predict_grid(self, fixture: FixtureFeatures) -> ProbabilityGrid:
        self.predictions += 1
        if self.model.fit_ is not None:
            try:
                return self.model.score_grid(
                    fixture.home_team_id, fixture.away_team_id
                )
            except (KeyError, ValueError, RuntimeError) as exc:
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
        self.fallback_predictions += 1
        home_rate = float(
            fixture.values.get("home_expected_goals_prior", self.fallback_home_rate)
        )
        away_rate = float(
            fixture.values.get("away_expected_goals_prior", self.fallback_away_rate)
        )
        return independent_poisson_grid(max(home_rate, 0.1), max(away_rate, 0.1))

    @property
    def fallback_rate(self) -> float:
        return self.fallback_predictions / self.predictions if self.predictions else 0.0
