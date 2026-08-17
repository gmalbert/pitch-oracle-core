"""A joint football score distribution with explicit truncation mass."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ProbabilityGrid:
    mass: np.ndarray
    tail_mass: float
    max_goals_home: int
    max_goals_away: int

    def __post_init__(self) -> None:
        values = np.asarray(self.mass, dtype=float)
        if values.ndim != 2 or values.size == 0:
            raise ValueError("mass must be a non-empty 2D array")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("mass must be finite and non-negative")
        if not np.isfinite(self.tail_mass) or not 0 <= self.tail_mass < 1:
            raise ValueError("tail_mass must be in [0, 1)")
        if values.shape != (self.max_goals_home + 1, self.max_goals_away + 1):
            raise ValueError("declared maxima do not match mass shape")
        if not np.isclose(values.sum() + self.tail_mass, 1.0, atol=1e-10):
            raise ValueError("represented mass plus tail must equal one")
        object.__setattr__(self, "mass", values)

    @property
    def represented_mass(self) -> float:
        return float(self.mass.sum())

    def outcome_lower_bounds(self) -> np.ndarray:
        return np.array(
            [
                np.tril(self.mass, k=-1).sum(),
                np.trace(self.mass),
                np.triu(self.mass, k=1).sum(),
            ],
            dtype=float,
        )

    def outcome_intervals(self) -> tuple[np.ndarray, np.ndarray]:
        lower = self.outcome_lower_bounds()
        return lower, np.minimum(1.0, lower + self.tail_mass)

    def normalized_one_x_two(self, max_tail: float = 1e-8) -> np.ndarray:
        if self.tail_mass > max_tail:
            raise ValueError(
                f"tail {self.tail_mass:.3g} exceeds point-market limit {max_tail:.3g}"
            )
        values = self.outcome_lower_bounds()
        return values / values.sum()

    def expected_goals_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        home = np.arange(self.mass.shape[0], dtype=float)[:, None]
        away = np.arange(self.mass.shape[1], dtype=float)[None, :]
        lower = (float((self.mass * home).sum()), float((self.mass * away).sum()))
        return lower, (float("inf"), float("inf"))

    def exact_score_probability(self, home_goals: int, away_goals: int) -> float:
        if home_goals < 0 or away_goals < 0:
            return 0.0
        if home_goals >= self.mass.shape[0] or away_goals >= self.mass.shape[1]:
            return 0.0
        return float(self.mass[home_goals, away_goals])

    def btts_lower_bound(self) -> float:
        return float(self.mass[1:, 1:].sum())

    def normalized_mass(self, max_tail: float = 1e-8) -> np.ndarray:
        if self.tail_mass > max_tail:
            raise ValueError("tail is too large to normalize as a point distribution")
        return self.mass / self.mass.sum()
