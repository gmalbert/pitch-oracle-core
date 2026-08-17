"""Nested-fold half-life selection without touching the forward partition."""

from collections.abc import Callable
import pandas as pd


def select_half_life(
    candidates: tuple[float, ...],
    folds: tuple[tuple[object, object], ...],
    evaluator: Callable[[float, pd.Timestamp, pd.Timestamp], float],
) -> tuple[float, pd.DataFrame]:
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("positive half-life candidates are required")
    rows = []
    for candidate in candidates:
        for fold_index, (cutoff, end) in enumerate(folds):
            score = evaluator(candidate, pd.Timestamp(cutoff), pd.Timestamp(end))
            rows.append({"half_life_days": candidate, "fold": fold_index, "score": score})
    report = pd.DataFrame(rows)
    selected = float(report.groupby("half_life_days").score.mean().idxmin())
    return selected, report
