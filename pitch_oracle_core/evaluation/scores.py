"""Proper probability scores in one canonical home/draw/away order."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
import numpy as np

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid

HOME, DRAW, AWAY = 0, 1, 2


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("probabilities must have shape (n, 3)")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("each probability row must sum to one")
    return values


def outcome_index(home_goals: np.ndarray, away_goals: np.ndarray) -> np.ndarray:
    home = np.asarray(home_goals)
    away = np.asarray(away_goals)
    if home.shape != away.shape:
        raise ValueError("home and away score vectors must have equal shape")
    return np.where(home > away, HOME, np.where(home == away, DRAW, AWAY)).astype(int)


def multiclass_log_loss(y: np.ndarray, probabilities: np.ndarray) -> float:
    p = validate_probabilities(probabilities)
    labels = np.asarray(y, dtype=int)
    if labels.shape != (len(p),) or not np.isin(labels, [HOME, DRAW, AWAY]).all():
        raise ValueError("invalid outcome labels")
    selected = np.clip(p[np.arange(len(p)), labels], 1e-15, 1.0)
    return float(-np.log(selected).mean())


def multiclass_brier(y: np.ndarray, probabilities: np.ndarray) -> float:
    p = validate_probabilities(probabilities)
    labels = np.asarray(y, dtype=int)
    if labels.shape != (len(p),) or not np.isin(labels, [HOME, DRAW, AWAY]).all():
        raise ValueError("invalid outcome labels")
    observed = np.eye(3, dtype=float)[labels]
    return float(np.square(p - observed).sum(axis=1).mean())


def ranked_probability_score(y: np.ndarray, probabilities: np.ndarray) -> float:
    p = validate_probabilities(probabilities)
    labels = np.asarray(y, dtype=int)
    if labels.shape != (len(p),) or not np.isin(labels, [HOME, DRAW, AWAY]).all():
        raise ValueError("invalid outcome labels")
    observed = np.eye(3, dtype=float)[labels]
    forecast_cdf = np.cumsum(p, axis=1)[:, :-1]
    observed_cdf = np.cumsum(observed, axis=1)[:, :-1]
    return float(np.square(forecast_cdf - observed_cdf).sum(axis=1).mean() / 2.0)


@dataclass(frozen=True)
class ScorePanel:
    fixtures: int
    log_loss: float
    brier: float
    rps: float
    accuracy: float
    draw_recall: float


def score_panel(y: np.ndarray, probabilities: np.ndarray) -> ScorePanel:
    p = validate_probabilities(probabilities)
    labels = np.asarray(y, dtype=int)
    prediction = p.argmax(axis=1)
    draw = labels == DRAW
    return ScorePanel(
        fixtures=len(labels),
        log_loss=multiclass_log_loss(labels, p),
        brier=multiclass_brier(labels, p),
        rps=ranked_probability_score(labels, p),
        accuracy=float((prediction == labels).mean()),
        draw_recall=float((prediction[draw] == DRAW).mean()) if draw.any() else float("nan"),
    )


def scoreline_ignorance(
    grid: ProbabilityGrid,
    observed_home: int,
    observed_away: int,
    *,
    floor: float = 1e-15,
) -> float:
    if floor <= 0:
        raise ValueError("floor must be positive")
    if observed_home <= grid.max_goals_home and observed_away <= grid.max_goals_away:
        probability = grid.exact_score_probability(observed_home, observed_away)
    else:
        probability = grid.tail_mass
    return -log(max(probability, floor))
