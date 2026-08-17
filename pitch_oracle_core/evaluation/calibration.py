"""Reproducible CORP-style reliability and calibration summaries."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReliabilityPoint:
    forecast_mean: float
    observed_rate: float
    fitted_rate: float
    count: int


def corp_reliability_curve(
    forecast_probability: np.ndarray, observed: np.ndarray
) -> list[ReliabilityPoint]:
    probability = np.asarray(forecast_probability, dtype=float)
    truth = np.asarray(observed, dtype=float)
    if probability.shape != truth.shape or probability.ndim != 1:
        raise ValueError("forecast and observed must be equal-length vectors")
    if (
        len(probability) < 2
        or not np.isfinite(probability).all()
        or (probability < 0).any()
        or (probability > 1).any()
    ):
        raise ValueError("invalid probability vector")
    if not np.isin(truth, [0.0, 1.0]).all():
        raise ValueError("observed values must be binary")
    order = np.argsort(probability, kind="mergesort")
    p_sorted, y_sorted = probability[order], truth[order]
    # Pool-adjacent-violators isotonic fit. Keeping this tiny implementation in
    # the evaluation layer prevents cached UI rendering from importing sklearn.
    blocks = [
        {"start": index, "end": index + 1, "sum": float(value), "count": 1}
        for index, value in enumerate(y_sorted)
    ]
    index = 0
    while index < len(blocks) - 1:
        left = blocks[index]["sum"] / blocks[index]["count"]
        right = blocks[index + 1]["sum"] / blocks[index + 1]["count"]
        if left <= right:
            index += 1
            continue
        blocks[index] = {
            "start": blocks[index]["start"],
            "end": blocks[index + 1]["end"],
            "sum": blocks[index]["sum"] + blocks[index + 1]["sum"],
            "count": blocks[index]["count"] + blocks[index + 1]["count"],
        }
        del blocks[index + 1]
        index = max(0, index - 1)
    fitted = np.empty(len(y_sorted), dtype=float)
    for block in blocks:
        fitted[block["start"]:block["end"]] = block["sum"] / block["count"]
    boundaries = np.r_[0, np.flatnonzero(np.diff(fitted) != 0) + 1, len(fitted)]
    return [
        ReliabilityPoint(
            forecast_mean=float(p_sorted[start:end].mean()),
            observed_rate=float(y_sorted[start:end].mean()),
            fitted_rate=float(fitted[start:end].mean()),
            count=int(end - start),
        )
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def expected_calibration_error(
    forecast_probability: np.ndarray, observed: np.ndarray, bins: int = 10
) -> float:
    if bins < 2:
        raise ValueError("bins must be at least two")
    p = np.asarray(forecast_probability, dtype=float)
    y = np.asarray(observed, dtype=float)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("forecast and observed must be equal-length vectors")
    index = np.minimum((p * bins).astype(int), bins - 1)
    result = 0.0
    for bucket in range(bins):
        selected = index == bucket
        if selected.any():
            result += selected.mean() * abs(p[selected].mean() - y[selected].mean())
    return float(result)
