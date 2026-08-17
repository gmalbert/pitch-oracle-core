"""Block-bootstrap paired deltas and a practical White-style reality check."""

from __future__ import annotations

import numpy as np


def circular_block_indices(
    observations: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    if observations < 2 or not 1 <= block_length <= observations:
        raise ValueError("invalid block dimensions")
    blocks = int(np.ceil(observations / block_length))
    starts = rng.integers(0, observations, size=blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % observations).ravel()[:observations]


def paired_block_interval(
    benchmark_loss: np.ndarray,
    candidate_loss: np.ndarray,
    *,
    block_length: int,
    repetitions: int = 5_000,
    seed: int = 20260810,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    benchmark = np.asarray(benchmark_loss, dtype=float)
    candidate = np.asarray(candidate_loss, dtype=float)
    if benchmark.shape != candidate.shape or benchmark.ndim != 1:
        raise ValueError("paired losses must be equal-length vectors")
    if repetitions < 1 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap configuration")
    advantage = benchmark - candidate
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample = circular_block_indices(len(advantage), block_length, rng)
        draws[index] = advantage[sample].mean()
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(advantage.mean()), float(lower), float(upper)


def white_style_reality_check(
    loss_advantages: np.ndarray,
    *,
    block_length: int,
    repetitions: int = 5_000,
    seed: int = 20260810,
) -> tuple[float, float]:
    values = np.asarray(loss_advantages, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("advantages must have shape (time, candidates)")
    n = values.shape[0]
    observed = float(np.sqrt(n) * np.max(values.mean(axis=0)))
    centered = values - values.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    null_statistics = np.empty(repetitions, dtype=float)
    for draw in range(repetitions):
        sample = circular_block_indices(n, block_length, rng)
        null_statistics[draw] = np.sqrt(n) * np.max(centered[sample].mean(axis=0))
    p_value = (1 + np.count_nonzero(null_statistics >= observed)) / (repetitions + 1)
    return observed, float(p_value)
