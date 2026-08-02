"""Risk assessment helpers for three-way match predictions.

Risk here measures how ambiguous the model's 1X2 probabilities are.  It does
not measure betting value; that requires market odds as well as probabilities.
"""

from __future__ import annotations

import math
from typing import Iterable


LOW_RISK_MAX = 60.0
MODERATE_RISK_MAX = 80.0
HIGH_RISK_MAX = 92.0


def calculate_prediction_risk(probabilities: Iterable[float]) -> tuple[float, float]:
    """Return ``(risk, confidence)`` for a three-outcome forecast.

    ``risk`` is on a 0-100 scale and ``confidence`` is on a 0-1 scale.  The
    score combines the leading outcome's probability with its margin over the
    runner-up, then normalizes the result so 33/33/33 is 100 risk and a certain
    outcome is 0 risk.

    Inputs are normalized to sum to one, which makes the helper tolerant of
    rounded percentages converted back to proportions.
    """
    probs = tuple(float(probability) for probability in probabilities)
    if len(probs) != 3:
        raise ValueError("Exactly three outcome probabilities are required")
    if any(not math.isfinite(probability) or probability < 0 for probability in probs):
        raise ValueError("Probabilities must be finite and non-negative")

    total = sum(probs)
    if total <= 0:
        raise ValueError("Probabilities must have a positive total")

    leader, runner_up, _ = sorted((probability / total for probability in probs), reverse=True)

    # The leader captures absolute certainty; the top-two margin prevents a
    # near tie from looking safe merely because both leading outcomes are high.
    raw_confidence = 0.70 * leader + 0.30 * (leader - runner_up)
    uniform_baseline = 0.70 / 3.0
    confidence = (raw_confidence - uniform_baseline) / (1.0 - uniform_baseline)
    confidence = min(1.0, max(0.0, confidence))
    return 100.0 * (1.0 - confidence), confidence


def get_risk_category(risk_score: float) -> tuple[str, str]:
    """Map a calibrated ambiguity score to a user-facing category and icon."""
    if risk_score <= LOW_RISK_MAX:
        return "Low Risk", "🟢"
    if risk_score <= MODERATE_RISK_MAX:
        return "Moderate Risk", "🟡"
    if risk_score <= HIGH_RISK_MAX:
        return "High Risk", "🟠"
    return "Critical Risk", "🔴"


def get_prediction_guidance(probabilities: Iterable[float], risk_score: float) -> tuple[str, str]:
    """Return cautious model guidance without claiming value absent market odds."""
    probs = tuple(float(probability) for probability in probabilities)
    if len(probs) != 3:
        raise ValueError("Exactly three outcome probabilities are required")
    if any(not math.isfinite(probability) or probability < 0 for probability in probs):
        raise ValueError("Probabilities must be finite and non-negative")
    total = sum(probs)
    if total <= 0:
        raise ValueError("Probabilities must have a positive total")
    probs = tuple(probability / total for probability in probs)

    labels = ("Home", "Draw", "Away")
    best_index = max(range(3), key=probs.__getitem__)
    max_prob = probs[best_index]

    if max_prob >= 0.65 and risk_score <= LOW_RISK_MAX:
        return f"Strong {labels[best_index]} Lean", "✅"
    if max_prob >= 0.55 and risk_score <= MODERATE_RISK_MAX:
        return f"Consider {labels[best_index]}", "🤔"
    return "No Clear Edge", "⏸️"
