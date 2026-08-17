"""Transparent feature, coverage, prediction, and performance drift."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("drift samples must contain finite values")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_share, _ = np.histogram(reference, bins=edges)
    current_share, _ = np.histogram(current, bins=edges)
    reference_share = np.clip(reference_share / reference_share.sum(), 1e-6, 1.0)
    current_share = np.clip(current_share / current_share.sum(), 1e-6, 1.0)
    return float(((current_share - reference_share) * np.log(
        current_share / reference_share
    )).sum())


def drift_severity(psi: float) -> str:
    if psi < 0:
        raise ValueError("PSI cannot be negative")
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "watch"
    return "action_required"


@dataclass(frozen=True)
class DriftFinding:
    category: str
    metric: str
    value: float
    severity: str
    evidence: str
    suggested_action: str
