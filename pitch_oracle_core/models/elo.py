"""Deterministic, point-in-time Elo strength baseline."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import log10


@dataclass
class EloModel:
    base_rating: float = 1500.0
    k_factor: float = 24.0
    home_advantage: float = 55.0
    season_regression: float = 0.25
    ratings: dict[str, float] = field(default_factory=dict)
    matches: dict[str, int] = field(default_factory=dict)
    history: dict[str, list[tuple[datetime, float]]] = field(default_factory=dict)

    def rating(self, team_id: str, prior: float | None = None) -> float:
        return self.ratings.get(team_id, self.base_rating if prior is None else prior)

    @staticmethod
    def expected(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def score(home_goals: int, away_goals: int) -> float:
        return 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0

    @staticmethod
    def goal_multiplier(goal_difference: int) -> float:
        difference = abs(goal_difference)
        return 1.0 if difference <= 1 else 1.0 + 0.5 * log10(float(difference))

    def update(
        self,
        home_team_id: str,
        away_team_id: str,
        home_goals: int,
        away_goals: int,
        *,
        home_prior: float | None = None,
        away_prior: float | None = None,
        observed_at: datetime | None = None,
    ) -> tuple[float, float]:
        if min(home_goals, away_goals) < 0:
            raise ValueError("goals cannot be negative")
        home = self.rating(home_team_id, home_prior)
        away = self.rating(away_team_id, away_prior)
        expected_home = self.expected(home + self.home_advantage, away)
        actual_home = self.score(home_goals, away_goals)
        change = self.k_factor * self.goal_multiplier(
            home_goals - away_goals
        ) * (actual_home - expected_home)
        self.ratings[home_team_id] = home + change
        self.ratings[away_team_id] = away - change
        self.matches[home_team_id] = self.matches.get(home_team_id, 0) + 1
        self.matches[away_team_id] = self.matches.get(away_team_id, 0) + 1
        if observed_at is not None:
            if observed_at.tzinfo is None:
                raise ValueError("observed_at must be timezone-aware")
            timestamp = observed_at.astimezone(timezone.utc)
            for team_id in (home_team_id, away_team_id):
                timeline = self.history.setdefault(team_id, [])
                if timeline and timestamp <= timeline[-1][0]:
                    raise ValueError("Elo updates must have strictly increasing timestamps")
                timeline.append((timestamp, self.ratings[team_id]))
        return self.ratings[home_team_id], self.ratings[away_team_id]

    def rating_at(self, team_id: str, cutoff_utc: datetime, prior: float | None = None) -> float:
        """Return the latest rating strictly observed before a forecast cutoff."""
        if cutoff_utc.tzinfo is None:
            raise ValueError("cutoff_utc must be timezone-aware")
        cutoff = cutoff_utc.astimezone(timezone.utc)
        candidates = [rating for observed, rating in self.history.get(team_id, ()) if observed < cutoff]
        return candidates[-1] if candidates else (self.base_rating if prior is None else prior)

    def regress_for_new_edition(self) -> None:
        for team_id, rating in self.ratings.items():
            self.ratings[team_id] = (
                (1.0 - self.season_regression) * rating
                + self.season_regression * self.base_rating
            )
