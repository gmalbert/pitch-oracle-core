"""Simulation-derived match stakes and points-target distributions."""

from __future__ import annotations

import math
import numpy as np

from .simulation import SimulationFixture, sample_score


def binary_entropy(probability: float) -> float:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if probability in (0.0, 1.0):
        return 0.0
    return -probability * math.log2(probability) - (1 - probability) * math.log2(1 - probability)


def match_stakes_index(
    base_outcomes: dict[str, float],
    conditional_outcomes: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Measure named-outcome movement conditional on home/draw/away results."""
    deltas: dict[str, dict[str, float]] = {}
    for result, probabilities in conditional_outcomes.items():
        deltas[result] = {
            label: probabilities.get(label, 0.0) - base_outcomes.get(label, 0.0)
            for label in set(base_outcomes) | set(probabilities)
        }
    magnitude = max(
        (abs(value) for result in deltas.values() for value in result.values()),
        default=0.0,
    )
    entropy_change = max(
        (
            abs(binary_entropy(probabilities.get(label, 0.0)) - binary_entropy(base))
            for label, base in base_outcomes.items()
            for probabilities in conditional_outcomes.values()
        ),
        default=0.0,
    )
    return {"index": float(max(magnitude, entropy_change)), "deltas": deltas}


def points_target_distribution(
    final_points: np.ndarray, final_positions: np.ndarray, target_position: int
) -> dict[str, float]:
    points = np.asarray(final_points, dtype=float)
    positions = np.asarray(final_positions, dtype=int)
    if points.shape != positions.shape or points.ndim != 2:
        raise ValueError("points and positions must be simulation-by-team matrices")
    qualifying = np.where(positions <= target_position, points, np.nan)
    threshold = np.nanmin(qualifying, axis=1)
    threshold = threshold[np.isfinite(threshold)]
    return {
        "p10": float(np.quantile(threshold, 0.10)),
        "median": float(np.quantile(threshold, 0.50)),
        "p90": float(np.quantile(threshold, 0.90)),
    }


def _conditional_score(
    fixture: SimulationFixture, outcome: str, uniform: float
) -> tuple[int, int]:
    matrix = np.asarray(fixture.score_matrix, dtype=float)
    home, away = np.indices(matrix.shape)
    masks = {
        "home": home > away,
        "draw": home == away,
        "away": home < away,
    }
    if outcome not in masks:
        raise ValueError("outcome must be home, draw, or away")
    weights = np.where(masks[outcome], matrix, 0.0).ravel()
    if weights.sum() <= 0:
        raise ValueError(f"focal fixture has no {outcome} score mass")
    cumulative = np.cumsum(weights / weights.sum())
    index = min(int(np.searchsorted(cumulative, uniform, side="right")), len(weights) - 1)
    return tuple(int(value) for value in np.unravel_index(index, matrix.shape))


def conditioned_match_stakes(
    rules,
    completed_matches,
    fixtures: list[SimulationFixture],
    *,
    focal_fixture_id: str,
    simulations: int = 10_000,
    seed: int = 20260810,
) -> dict[str, object]:
    """Estimate H/D/A stakes with common random numbers for remaining fixtures."""
    if simulations < 1:
        raise ValueError("simulations must be positive")
    focal = next(
        (fixture for fixture in fixtures if fixture.fixture_id == focal_fixture_id), None
    )
    if focal is None:
        raise KeyError(f"Unknown focal fixture: {focal_fixture_id}")
    remaining = [fixture for fixture in fixtures if fixture.fixture_id != focal_fixture_id]
    matrix = np.asarray(focal.score_matrix, dtype=float)
    home, away = np.indices(matrix.shape)
    outcome_weights = {
        "home": float(matrix[home > away].sum()),
        "draw": float(matrix[home == away].sum()),
        "away": float(matrix[home < away].sum()),
    }
    master = np.random.default_rng(seed)
    scenario_seeds = master.integers(0, np.iinfo(np.int64).max, size=simulations)
    focal_uniforms = master.random(simulations)
    counts: dict[str, dict[str, int]] = {
        outcome: {} for outcome in outcome_weights
    }
    for simulation_index, scenario_seed in enumerate(scenario_seeds):
        for outcome in outcome_weights:
            # Resetting to the same per-simulation seed is the common-random-number
            # guarantee: only the focal result differs across the three worlds.
            rng = np.random.default_rng(int(scenario_seed))
            state = rules.initial_state(completed_matches)
            focal_score = _conditional_score(
                focal, outcome, float(focal_uniforms[simulation_index])
            )
            rules.apply_score(state, focal, *focal_score)
            queue = list(remaining)
            while not rules.is_complete(state):
                if not queue:
                    queue.extend(rules.next_fixtures(state))
                    if not queue:
                        raise RuntimeError("Rules produced no fixtures for incomplete season")
                fixture = queue.pop(0)
                score = sample_score(fixture.score_matrix, rng)
                rules.apply_score(state, fixture, *score)
            for label, teams in rules.outcome_labels(state).items():
                for team_id in teams:
                    key = f"{team_id}:{label}"
                    counts[outcome][key] = counts[outcome].get(key, 0) + 1
    conditional = {
        outcome: {
            key: count / simulations for key, count in outcome_counts.items()
        }
        for outcome, outcome_counts in counts.items()
    }
    all_keys = set().union(*(values.keys() for values in conditional.values()))
    base = {
        key: sum(
            outcome_weights[outcome] * conditional[outcome].get(key, 0.0)
            for outcome in outcome_weights
        )
        for key in all_keys
    }
    report = match_stakes_index(base, conditional)
    return {
        **report,
        "focal_fixture_id": focal_fixture_id,
        "simulations": simulations,
        "seed": seed,
        "common_random_numbers": True,
        "base_outcomes": base,
        "conditional_outcomes": conditional,
        "outcome_weights": outcome_weights,
    }
