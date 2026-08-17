"""Bootstrap forecast intervals and empirical coverage."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ForecastInterval:
    median: np.ndarray
    lower_50: np.ndarray
    upper_50: np.ndarray
    lower_80: np.ndarray
    upper_80: np.ndarray


def probability_intervals(draws: np.ndarray) -> ForecastInterval:
    values = np.asarray(draws, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("Expected draws with shape (draw, fixture, 3)")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Bootstrap probabilities must be finite and non-negative")
    totals = values.sum(axis=2, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Each bootstrap probability row needs positive mass")
    values = values / totals
    return ForecastInterval(
        median=np.quantile(values, 0.50, axis=0),
        lower_50=np.quantile(values, 0.25, axis=0),
        upper_50=np.quantile(values, 0.75, axis=0),
        lower_80=np.quantile(values, 0.10, axis=0),
        upper_80=np.quantile(values, 0.90, axis=0),
    )


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    targets = np.eye(3)[np.asarray(y_true, dtype=int)]
    return ((targets >= lower) & (targets <= upper)).mean(axis=0)


def leader_stability(draws: np.ndarray) -> np.ndarray:
    values = np.asarray(draws, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("Expected draws with shape (draw, fixture, 3)")
    point_leader = np.median(values, axis=0).argmax(axis=1)
    return np.mean(values.argmax(axis=2) == point_leader[None, :], axis=0)
