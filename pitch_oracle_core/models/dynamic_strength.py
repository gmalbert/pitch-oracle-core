"""Simple score-driven latent strength state for R08 experiments."""

from dataclasses import dataclass, field
import math


@dataclass
class DynamicStrengthModel:
    learning_rate: float = 0.05
    regression: float = 0.01
    strengths: dict[str, float] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)

    def predict_home_probability(
        self, home_team_id: str, away_team_id: str, home_advantage: float = 0.15
    ) -> float:
        difference = self.strengths.get(home_team_id, 0.0) - self.strengths.get(away_team_id, 0.0)
        return 1 / (1 + math.exp(-(difference + home_advantage)))

    def update(
        self, home_team_id: str, away_team_id: str, home_goals: int, away_goals: int
    ) -> None:
        expected = self.predict_home_probability(home_team_id, away_team_id)
        actual = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        innovation = actual - expected
        for team_id, sign in ((home_team_id, 1.0), (away_team_id, -1.0)):
            old = self.strengths.get(team_id, 0.0)
            self.strengths[team_id] = (1 - self.regression) * old + sign * self.learning_rate * innovation
            self.uncertainty[team_id] = max(0.05, self.uncertainty.get(team_id, 1.0) * 0.98)
