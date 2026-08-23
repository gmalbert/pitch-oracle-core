"""Cohort performance slices — promoted teams, derbies, short rest, early season.

Extends the existing ``evaluation.cohorts`` with richer slicing for
diagnostic reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from penaltyblog.metrics import ignorance_score, multiclass_brier_score, rps_average


def assign_cohorts(
    evaluations: pd.DataFrame,
    *,
    rest_days_col: str = "rest_days",
    is_promoted_col: str = "is_promoted",
    is_derby_col: str = "is_derby",
    matchday_col: str = "matchday",
) -> pd.DataFrame:
    """Tag each evaluation row with cohort labels."""
    df = evaluations.copy()
    # Rest days cohort
    if rest_days_col in df.columns:
        df["cohort_rest"] = pd.cut(
            df[rest_days_col], bins=[0, 3, 5, 7, 14, 100],
            labels=["short_rest", "normal", "extended", "long_break", "very_long"],
        )
    else:
        df["cohort_rest"] = "unknown"
    # Promoted cohort
    if is_promoted_col in df.columns:
        df["cohort_promoted"] = df[is_promoted_col].map({True: "promoted", False: "established"})
    else:
        df["cohort_promoted"] = "unknown"
    # Derby cohort
    if is_derby_col in df.columns:
        df["cohort_derby"] = df[is_derby_col].map({True: "derby", False: "non_derby"})
    else:
        df["cohort_derby"] = "unknown"
    # Season phase
    if matchday_col in df.columns:
        df["cohort_phase"] = pd.cut(
            df[matchday_col], bins=[0, 10, 25, 38, 100],
            labels=["early", "mid", "late", "post"],
        )
    else:
        df["cohort_phase"] = "unknown"
    return df


def cohort_performance(
    evaluations: pd.DataFrame,
    cohort_col: str,
    *,
    outcome_col: str = "result",
    prob_home_col: str = "p_home",
    prob_draw_col: str = "p_draw",
    prob_away_col: str = "p_away",
    minimum_sample: int = 30,
) -> pd.DataFrame:
    """Compute proper scores per cohort slice."""
    outcome_map = {"H": 0, "D": 1, "A": 2}
    rows = []
    for cohort, group in evaluations.groupby(cohort_col):
        if len(group) < minimum_sample:
            continue
        valid = group[group[outcome_col].isin(outcome_map)]
        if len(valid) < minimum_sample:
            continue
        y = valid[outcome_col].map(outcome_map).to_numpy()
        p = valid[[prob_home_col, prob_draw_col, prob_away_col]].to_numpy()
        rows.append({
            "cohort": cohort,
            "n": len(valid),
            "brier": float(multiclass_brier_score(p, y)),
            "log_loss": float(ignorance_score(p, y)),
            "rps": float(rps_average(p, y)),
        })
    return pd.DataFrame(rows).sort_values("brier")


def full_cohort_report(
    evaluations: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Run cohort analysis across all cohort dimensions."""
    df = assign_cohorts(evaluations, **kwargs)
    frames = []
    for col in ("cohort_rest", "cohort_promoted", "cohort_derby", "cohort_phase"):
        if col in df.columns:
            report = cohort_performance(df, col)
            report["dimension"] = col
            frames.append(report)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
