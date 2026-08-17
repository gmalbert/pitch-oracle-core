"""Interpretable Pi/Glicko-lite candidates for a frozen rating tournament."""

from dataclasses import dataclass, field
import math


@dataclass
class GlickoLite:
    ratings: dict[str, float] = field(default_factory=dict)
    deviations: dict[str, float] = field(default_factory=dict)
    base: float = 1500.0

    def expected(self, home: str, away: str, home_advantage: float = 55.0) -> float:
        h, a = self.ratings.get(home, self.base), self.ratings.get(away, self.base)
        return 1 / (1 + 10 ** ((a - h - home_advantage) / 400))

    def update(self, home: str, away: str, score: float) -> None:
        expected = self.expected(home, away)
        for team, sign in ((home, 1), (away, -1)):
            deviation = self.deviations.get(team, 350.0)
            k = max(8.0, min(32.0, deviation / 12))
            self.ratings[team] = self.ratings.get(team, self.base) + sign * k * (score - expected)
            self.deviations[team] = max(60.0, deviation * 0.97)


@dataclass
class PiRating:
    ratings: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.04

    def update(self, home: str, away: str, goal_difference: int) -> None:
        predicted = self.ratings.get(home, 0.0) - self.ratings.get(away, 0.0)
        error = math.copysign(math.log1p(abs(goal_difference - predicted)), goal_difference - predicted)
        self.ratings[home] = self.ratings.get(home, 0.0) + self.learning_rate * error
        self.ratings[away] = self.ratings.get(away, 0.0) - self.learning_rate * error
