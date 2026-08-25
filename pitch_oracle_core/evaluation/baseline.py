"""Metrics and calibration baseline backed by penaltyblog.metrics.

Provides proper scoring rules (Brier, log-loss, RPS) and calibration
diagnostics (reliability table, ECE/MCE) for the goal-model pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from penaltyblog.metrics import ignorance_score, multiclass_brier_score, rps_average

from .calibration import expected_calibration_error


def validate_outcome_array(y: np.ndarray) -> np.ndarray:
    labels = np.asarray(y, dtype=int)
    if labels.ndim != 1 or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("outcomes must be a 1D array of 0 (home), 1 (draw), 2 (away)")
    return labels


def validate_probability_array(p: np.ndarray) -> np.ndarray:
    values = np.asarray(p, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("probabilities must have shape (n, 3)")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each probability row must sum to one")
    return values


def proper_score_summary(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Compute the three canonical proper scores via penaltyblog.metrics.

    Returns a dict with keys ``brier``, ``log_loss``, ``rps``, and ``fixtures``.
    """
    labels = validate_outcome_array(y)
    p = validate_probability_array(probabilities)
    return {
        "fixtures": len(labels),
        "brier": float(multiclass_brier_score(p, labels)),
        "log_loss": float(ignorance_score(p, labels)),
        "rps": float(rps_average(p, labels)),
    }


def reliability_table(
    forecast_probability: np.ndarray,
    observed: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Decile-binned reliability table for a binary outcome.

    Returns a DataFrame with columns:
    ``bin_midpoint``, ``n``, ``fraction_positive``, ``predicted``, ``gap``.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    p = np.asarray(forecast_probability, dtype=float)
    y = np.asarray(observed, dtype=float)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("forecast and observed must be equal-length vectors")
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.minimum(np.digitize(p, bin_edges[1:-1], right=True), n_bins - 1)
    rows = []
    for bucket in range(n_bins):
        selected = bin_indices == bucket
        count = int(selected.sum())
        if count == 0:
            rows.append({
                "bin_midpoint": round((bin_edges[bucket] + bin_edges[bucket + 1]) / 2, 4),
                "n": 0, "fraction_positive": np.nan,
                "predicted": np.nan, "gap": np.nan,
            })
        else:
            predicted_mean = float(p[selected].mean())
            observed_mean = float(y[selected].mean())
            rows.append({
                "bin_midpoint": round((bin_edges[bucket] + bin_edges[bucket + 1]) / 2, 4),
                "n": count,
                "fraction_positive": round(observed_mean, 6),
                "predicted": round(predicted_mean, 6),
                "gap": round(abs(predicted_mean - observed_mean), 6),
            })
    return pd.DataFrame(rows)


def calibration_error_summary(
    forecast_probability: np.ndarray,
    observed: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Expected and maximum calibration error in decile bins."""
    ece = expected_calibration_error(forecast_probability, observed, bins=n_bins)
    p = np.asarray(forecast_probability, dtype=float)
    y = np.asarray(observed, dtype=float)
    index = np.minimum((p * n_bins).astype(int), n_bins - 1)
    mce = 0.0
    for bucket in range(n_bins):
        selected = index == bucket
        if selected.any():
            mce = max(mce, abs(p[selected].mean() - y[selected].mean()))
    return {"ece": float(ece), "mce": float(mce)}
