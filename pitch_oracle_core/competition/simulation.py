"""Rule-adapter season simulation over coherent fixture score grids."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SimulationFixture:
    fixture_id: str
    home_team_id: str
    away_team_id: str
    score_matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.score_matrix, dtype=float)
        if matrix.ndim != 2 or (matrix < 0).any() or not np.isfinite(matrix).all():
            raise ValueError("simulation score matrix is invalid")
        if not np.isclose(matrix.sum(), 1.0, atol=1e-8):
            raise ValueError("simulation score matrix must sum to one")


class CompetitionRuleAdapter:
    def initial_state(self, completed_matches): ...
    def apply_score(self, state, fixture, home_goals: int, away_goals: int): ...
    def next_fixtures(self, state) -> list[SimulationFixture]: ...
    def is_complete(self, state) -> bool: ...
    def ranked_teams(self, state) -> list[str]: ...
    def outcome_labels(self, state) -> dict[str, set[str]]: ...


def sample_score(matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flattened = np.asarray(matrix, dtype=float).ravel()
    if not np.isfinite(flattened).all() or (flattened < 0).any() or flattened.sum() <= 0:
        raise ValueError("cannot sample an invalid score matrix")
    flattened /= flattened.sum()
    index = int(rng.choice(len(flattened), p=flattened))
    return tuple(int(value) for value in np.unravel_index(index, matrix.shape))


def simulate_season(
    rules: CompetitionRuleAdapter,
    completed_matches,
    initial_fixtures: list[SimulationFixture],
    *,
    simulations: int = 20_000,
    seed: int = 20260810,
) -> dict[str, object]:
    if simulations < 1:
        raise ValueError("simulations must be positive")
    rng = np.random.default_rng(seed)
    positions: dict[str, Counter[int]] = defaultdict(Counter)
    outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    points: dict[str, list[float]] = defaultdict(list)
    for _ in range(simulations):
        state = rules.initial_state(completed_matches)
        queue = list(initial_fixtures)
        while not rules.is_complete(state):
            if not queue:
                queue.extend(rules.next_fixtures(state))
                if not queue:
                    raise RuntimeError("Rules produced no fixtures for incomplete season")
            fixture = queue.pop(0)
            home_goals, away_goals = sample_score(fixture.score_matrix, rng)
            rules.apply_score(state, fixture, home_goals, away_goals)
        ranking = rules.ranked_teams(state)
        labels = rules.outcome_labels(state)
        for position, team_id in enumerate(ranking, start=1):
            positions[team_id][position] += 1
        for label, members in labels.items():
            for team_id in members:
                outcomes[team_id][label] += 1
        state_points = getattr(state, "points", None)
        if isinstance(state_points, dict):
            for team_id, value in state_points.items():
                points[team_id].append(float(value))
    return {
        "simulations": simulations,
        "position_probabilities": {
            team_id: {
                position: count / simulations for position, count in counts.items()
            }
            for team_id, counts in positions.items()
        },
        "outcome_probabilities": {
            team_id: {label: count / simulations for label, count in counts.items()}
            for team_id, counts in outcomes.items()
        },
        "expected_points": {
            team_id: float(np.mean(values)) for team_id, values in points.items()
        },
    }


def validate_simulation_probabilities(result: dict[str, object], team_count: int) -> None:
    positions = result["position_probabilities"]
    for team_id, probabilities in positions.items():
        if not np.isclose(sum(probabilities.values()), 1.0, atol=1e-8):
            raise ValueError(f"Position probabilities do not sum to one for {team_id}")
    for position in range(1, team_count + 1):
        total = sum(values.get(position, 0.0) for values in positions.values())
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError(f"Position {position} probabilities do not sum to one")
