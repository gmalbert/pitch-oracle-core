"""Odds conversion pipeline — American, fractional, and decimal across the ingest.

Wraps ``penaltyblog.betting.convert_odds`` with convenience helpers for
batch conversion and format detection.
"""

from __future__ import annotations

from penaltyblog.betting import convert_odds


def decimal_to_american(decimal_odds: float) -> int | None:
    """Convert decimal odds to American format."""
    if decimal_odds <= 1:
        return None
    return round((decimal_odds - 1) * 100) if decimal_odds >= 2 else round(-100 / (decimal_odds - 1))


def american_to_decimal(american_odds: float) -> float | None:
    """Convert American odds to decimal format."""
    if american_odds == 0:
        return None
    return 1.0 + (american_odds / 100) if american_odds > 0 else 1.0 + (100 / abs(american_odds))


def fractional_to_decimal(fractional: str) -> float | None:
    """Convert fractional odds string (e.g. '5/2') to decimal."""
    try:
        num, den = fractional.split("/")
        return 1.0 + float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        return None


def convert_odds_batch(
    values: list[float | str],
    from_format: str = "decimal",
    to_format: str = "decimal",
) -> list[float]:
    """Batch convert odds between formats using penaltyblog's engine."""
    return [convert_odds(v, from_format=from_format, to_format=to_format) for v in values]


def detect_odds_format(value: float | str) -> str:
    """Heuristic detection of odds format."""
    if isinstance(value, str) and "/" in value:
        return "fractional"
    if isinstance(value, (int, float)):
        if value > 100 or value < -100:
            return "american"
        if value > 1:
            return "decimal"
    return "unknown"
