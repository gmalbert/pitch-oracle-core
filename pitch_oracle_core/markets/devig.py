"""Multiplicative, power, and Shin de-vig conversions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import numpy as np


class DevigMethod(StrEnum):
    MULTIPLICATIVE = "multiplicative"
    POWER = "power"
    SHIN = "shin"


@dataclass(frozen=True)
class FairMarket:
    probabilities: np.ndarray
    method: DevigMethod
    overround: float
    parameter: float | None

    def __post_init__(self) -> None:
        p = np.asarray(self.probabilities, dtype=float)
        if p.ndim != 1 or (p <= 0).any() or not np.isclose(p.sum(), 1.0, atol=1e-10):
            raise ValueError("fair probabilities must be positive and sum to one")
        object.__setattr__(self, "probabilities", p)


def inverse_odds(decimal_odds: np.ndarray) -> np.ndarray:
    odds = np.asarray(decimal_odds, dtype=float)
    if (
        odds.ndim != 1
        or len(odds) < 2
        or not np.isfinite(odds).all()
        or (odds <= 1).any()
    ):
        raise ValueError("decimal odds must be a vector strictly above one")
    return 1.0 / odds


def _bisect_root(function, low: float, high: float, iterations: int = 200) -> float:
    f_low, f_high = function(low), function(high)
    if f_low == 0:
        return low
    if f_high == 0:
        return high
    if f_low * f_high > 0:
        raise ValueError("root is not bracketed")
    for _ in range(iterations):
        middle = (low + high) / 2
        f_middle = function(middle)
        if abs(f_middle) < 1e-13:
            return middle
        if f_low * f_middle <= 0:
            high, f_high = middle, f_middle
        else:
            low, f_low = middle, f_middle
    return (low + high) / 2


def devig(decimal_odds: np.ndarray, method: DevigMethod | str) -> FairMarket:
    method = DevigMethod(method)
    inverse = inverse_odds(decimal_odds)
    total = float(inverse.sum())
    overround = total - 1.0
    if overround < -1e-12:
        raise ValueError("underround markets require an explicit policy")
    if abs(overround) <= 1e-12:
        return FairMarket(inverse / total, method, max(0.0, overround), None)
    if method == DevigMethod.MULTIPLICATIVE:
        return FairMarket(inverse / total, method, overround, None)
    if method == DevigMethod.POWER:
        objective = lambda exponent: float(np.power(inverse, exponent).sum() - 1.0)
        exponent = _bisect_root(objective, 1.0, 100.0)
        probability = np.power(inverse, exponent)
        return FairMarket(probability / probability.sum(), method, overround, exponent)
    if method == DevigMethod.SHIN:
        def probabilities(z: float) -> np.ndarray:
            numerator = np.sqrt(
                z * z + 4.0 * (1.0 - z) * np.square(inverse) / total
            ) - z
            return numerator / (2.0 * (1.0 - z))

        objective = lambda z: float(probabilities(z).sum() - 1.0)
        z = _bisect_root(objective, 0.0, 1.0 - 1e-12)
        probability = probabilities(z)
        return FairMarket(probability / probability.sum(), method, overround, z)
    raise ValueError(f"unsupported de-vig method: {method}")
