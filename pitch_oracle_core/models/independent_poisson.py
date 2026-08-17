"""Independent Poisson baseline implementing the explicit-tail grid contract."""

from math import exp, lgamma, log
import numpy as np

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid


def poisson_pmf(goals: int, rate: float) -> float:
    if goals < 0 or rate <= 0 or not np.isfinite(rate):
        raise ValueError("goals must be non-negative and rate finite/positive")
    return exp(goals * log(rate) - rate - lgamma(goals + 1))


def independent_poisson_grid(
    home_rate: float,
    away_rate: float,
    *,
    tail_tolerance: float = 1e-10,
    initial_max_goals: int = 8,
    hard_max_goals: int = 30,
) -> ProbabilityGrid:
    if not 0 <= tail_tolerance < 1:
        raise ValueError("tail_tolerance must be in [0, 1)")
    if initial_max_goals < 0 or hard_max_goals < initial_max_goals:
        raise ValueError("invalid score-grid bounds")
    for maximum in range(initial_max_goals, hard_max_goals + 1):
        home = np.array([poisson_pmf(g, home_rate) for g in range(maximum + 1)])
        away = np.array([poisson_pmf(g, away_rate) for g in range(maximum + 1)])
        mass = np.outer(home, away)
        represented = float(mass.sum())
        tail = max(0.0, 1.0 - represented)
        if tail <= tail_tolerance:
            if represented > 1.0:
                mass = mass / represented
                tail = 0.0
            else:
                tail = 1.0 - represented
            return ProbabilityGrid(mass, tail, maximum, maximum)
    raise RuntimeError("score grid did not meet tail tolerance before hard maximum")
