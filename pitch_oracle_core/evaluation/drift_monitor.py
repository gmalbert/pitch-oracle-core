"""Drift monitor MVP — Population Stability Index on features.

Detects when the distribution of input features has shifted significantly
from a reference window, triggering alerts before model quality degrades.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftFinding:
    """A single feature's drift assessment."""
    feature: str
    psi: float
    severity: str  # "stable", "watch", "action_required"


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute PSI between reference and current distributions.

    PSI < 0.1: stable, 0.1–0.25: watch, > 0.25: action_required.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.linspace(min(ref.min(), cur.min()), max(ref.max(), cur.max()), bins + 1)
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float) / len(ref)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float) / len(cur)
    # Avoid log(0)
    ref_counts = np.clip(ref_counts, 1e-6, None)
    cur_counts = np.clip(cur_counts, 1e-6, None)
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def drift_severity(psi: float) -> str:
    """Classify PSI into severity levels."""
    if psi > 0.25:
        return "action_required"
    if psi > 0.1:
        return "watch"
    return "stable"


def monitor_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = 0.25,
) -> list[DriftFinding]:
    """Check PSI for each feature between reference and current DataFrames."""
    if features is None:
        features = [c for c in reference.columns if reference[c].dtype in (np.float64, np.int64)]
    findings = []
    for feature in features:
        if feature not in current.columns:
            continue
        ref_vals = reference[feature].dropna().values
        cur_vals = current[feature].dropna().values
        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue
        psi = population_stability_index(ref_vals, cur_vals)
        findings.append(DriftFinding(
            feature=feature,
            psi=psi,
            severity=drift_severity(psi),
        ))
    return findings


def drift_report(findings: list[DriftFinding]) -> pd.DataFrame:
    """Convert drift findings to a summary DataFrame."""
    return pd.DataFrame([
        {"feature": f.feature, "psi": round(f.psi, 4), "severity": f.severity}
        for f in sorted(findings, key=lambda x: x.psi, reverse=True)
    ])
