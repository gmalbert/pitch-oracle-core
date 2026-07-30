"""Small, deterministic shot-based xG proxy for non-Understat leagues."""

from dataclasses import dataclass
from math import exp
from typing import Iterable


@dataclass(frozen=True)
class Shot:
    distance: float
    angle: float = 0.0
    header: bool = False
    body_part: str = "foot"


def expected_goals_from_shots(shots: Iterable[Shot]) -> float:
    """Estimate xG from shot descriptors; missing/invalid shots contribute zero."""
    total = 0.0
    for shot in shots:
        if shot.distance < 0:
            raise ValueError("shot distance cannot be negative")
        logit = 1.65 - 0.105 * shot.distance + 0.012 * shot.angle
        if shot.header:
            logit -= 0.45
        if shot.body_part.lower() == "foot":
            logit += 0.15
        total += 1 / (1 + exp(-logit))
    return total

