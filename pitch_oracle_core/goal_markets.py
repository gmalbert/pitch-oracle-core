"""Goal-market probabilities derived from independent Poisson goal rates.

The functions in this module are deliberately dependency-light so every league
consumer can use the same market contract, regardless of its xG data provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .models.independent_poisson import independent_poisson_grid


@dataclass(frozen=True)
class GoalMarketProbabilities:
    """Probabilities for common goal-based football markets."""

    home_expected_goals: float
    away_expected_goals: float
    over_under: dict[float, tuple[float, float]]
    btts_yes: float
    btts_no: float
    most_likely_score: tuple[int, int]

    @property
    def total_expected_goals(self) -> float:
        return self.home_expected_goals + self.away_expected_goals

    def as_dict(self) -> dict[str, object]:
        """Return a stable, JSON-friendly output contract."""
        result: dict[str, object] = {
            "ExpectedHomeGoals": round(self.home_expected_goals, 4),
            "ExpectedAwayGoals": round(self.away_expected_goals, 4),
            "ExpectedTotalGoals": round(self.total_expected_goals, 4),
            "OverUnderProbabilities": {
                line: {"over": round(over, 4), "under": round(under, 4)}
                for line, (over, under) in self.over_under.items()
            },
            "BTTSProb": round(self.btts_yes, 4),
            "BTTSYesProb": round(self.btts_yes, 4),
            "BTTSNoProb": round(self.btts_no, 4),
            "MostLikelyScore": f"{self.most_likely_score[0]}-{self.most_likely_score[1]}",
        }
        for line, (over, under) in self.over_under.items():
            suffix = str(line).replace(".", "_")
            result[f"Over{suffix}Prob"] = round(over, 4)
            result[f"Under{suffix}Prob"] = round(under, 4)
        return result


def _validate_expected_goals(home_expected_goals: float, away_expected_goals: float) -> None:
    if not isfinite(home_expected_goals) or not isfinite(away_expected_goals):
        raise ValueError("expected goals must be finite")
    if home_expected_goals < 0 or away_expected_goals < 0:
        raise ValueError("expected goals cannot be negative")


def calculate_goal_markets(
    home_expected_goals: float,
    away_expected_goals: float,
    *,
    lines: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5),
    max_goals: int = 10,
) -> GoalMarketProbabilities:
    """Calculate scoreline, O/U, and BTTS probabilities from expected goals.

    Expected goals are treated as independent Poisson rates. O/U probabilities
    use the exact Poisson distribution of total goals, while the scoreline is
    searched through ``max_goals`` for display purposes.
    """
    _validate_expected_goals(home_expected_goals, away_expected_goals)
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")

    grid = independent_poisson_grid(
        home_expected_goals,
        away_expected_goals,
        initial_max_goals=max(8, max_goals),
    )
    matrix = grid.normalized_mass()
    home_goals = np.arange(matrix.shape[0])[:, None]
    away_goals = np.arange(matrix.shape[1])[None, :]
    total_goals = home_goals + away_goals
    over_under = {}
    for line in lines:
        if line < 0 or line % 1 != 0.5:
            raise ValueError("goal lines must be non-negative half-goal values, e.g. 2.5")
        under = float(matrix[total_goals < line].sum())
        over_under[line] = (float(matrix[total_goals > line].sum()), under)
    btts_yes = float(matrix[1:, 1:].sum())
    best_score = tuple(
        int(value) for value in np.unravel_index(int(matrix.argmax()), matrix.shape)
    )

    return GoalMarketProbabilities(
        home_expected_goals=home_expected_goals,
        away_expected_goals=away_expected_goals,
        over_under=over_under,
        btts_yes=btts_yes,
        btts_no=1.0 - btts_yes,
        most_likely_score=best_score,
    )
