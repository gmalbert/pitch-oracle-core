"""Frozen score-distribution candidate families and diagnostic-gated grids."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial, lgamma, log
import numpy as np

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid
from .independent_poisson import independent_poisson_grid, poisson_pmf


@dataclass(frozen=True)
class DistributionCandidate:
    family: str
    status: str
    diagnostic_gate: str


CANDIDATES = (
    DistributionCandidate("independent_poisson", "baseline", "always"),
    DistributionCandidate("dixon_coles", "challenger", "low-score residual"),
    DistributionCandidate("bivariate_poisson", "challenger", "score covariance"),
    DistributionCandidate("diagonal_inflated", "challenger", "diagonal residual"),
    DistributionCandidate("negative_binomial", "challenger", "overdispersion"),
    DistributionCandidate("com_poisson", "challenger", "under/overdispersion"),
    DistributionCandidate("weibull_copula", "deferred", "material forward gain"),
)


def _grid_from_mass(mass: np.ndarray) -> ProbabilityGrid:
    represented = float(mass.sum())
    if represented <= 0 or represented > 1 + 1e-10:
        raise ValueError("candidate generated invalid represented mass")
    if represented > 1:
        mass = mass / represented
        represented = 1.0
    return ProbabilityGrid(mass, 1.0 - represented, mass.shape[0] - 1, mass.shape[1] - 1)


def bivariate_poisson_grid(
    home_rate: float,
    away_rate: float,
    shared_rate: float,
    *,
    max_goals: int = 14,
) -> ProbabilityGrid:
    if min(home_rate, away_rate) <= shared_rate or shared_rate < 0:
        raise ValueError("shared rate must be non-negative and below both marginal means")
    first, second = home_rate - shared_rate, away_rate - shared_rate
    mass = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    constant = exp(-(first + second + shared_rate))
    for home in range(max_goals + 1):
        for away in range(max_goals + 1):
            total = 0.0
            for shared in range(min(home, away) + 1):
                total += (
                    first ** (home - shared)
                    * second ** (away - shared)
                    * shared_rate ** shared
                    / (
                        factorial(home - shared)
                        * factorial(away - shared)
                        * factorial(shared)
                    )
                )
            mass[home, away] = constant * total
    return _grid_from_mass(mass)


def diagonal_inflated_grid(
    base: ProbabilityGrid, inflation: float
) -> ProbabilityGrid:
    if not -0.9 < inflation < 10:
        raise ValueError("diagonal inflation is outside the stable region")
    mass = base.mass.copy()
    diagonal = np.eye(*mass.shape, dtype=bool)
    mass[diagonal] *= 1 + inflation
    target = base.represented_mass
    mass *= target / mass.sum()
    return ProbabilityGrid(
        mass, base.tail_mass, base.max_goals_home, base.max_goals_away
    )


def negative_binomial_pmf(goals: int, mean: float, dispersion: float) -> float:
    if goals < 0 or mean <= 0 or dispersion <= 0:
        raise ValueError("invalid negative-binomial parameters")
    size = dispersion
    probability = size / (size + mean)
    return exp(
        lgamma(goals + size) - lgamma(size) - lgamma(goals + 1)
        + size * log(probability) + goals * log(1 - probability)
    )


def negative_binomial_grid(
    home_mean: float,
    away_mean: float,
    dispersion: float,
    *,
    max_goals: int = 20,
) -> ProbabilityGrid:
    home = np.array([
        negative_binomial_pmf(goal, home_mean, dispersion)
        for goal in range(max_goals + 1)
    ])
    away = np.array([
        negative_binomial_pmf(goal, away_mean, dispersion)
        for goal in range(max_goals + 1)
    ])
    return _grid_from_mass(np.outer(home, away))


def com_poisson_marginal(rate: float, dispersion: float, max_goals: int) -> np.ndarray:
    if rate <= 0 or dispersion <= 0:
        raise ValueError("CMP parameters must be positive")
    log_weights = np.array([
        goal * log(rate) - dispersion * lgamma(goal + 1)
        for goal in range(max_goals + 1)
    ])
    weights = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()


def com_poisson_grid(
    home_rate: float,
    away_rate: float,
    dispersion: float,
    *,
    max_goals: int = 20,
) -> ProbabilityGrid:
    home = com_poisson_marginal(home_rate, dispersion, max_goals)
    away = com_poisson_marginal(away_rate, dispersion, max_goals)
    mass = np.outer(home, away)
    return ProbabilityGrid(mass, 0.0, max_goals, max_goals)
