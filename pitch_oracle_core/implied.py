"""Canonical implied-probability conversions backed by penaltyblog.implied.

Every call routes through ``penaltyblog.implied.calculate_implied`` with the
``LOGARITHMIC`` method as the default — the most robust de-vig for 1X2 markets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from penaltyblog.implied import (
    ImpliedMethod,
    ImpliedProbabilities,
    OddsFormat,
    calculate_implied,
)

DEFAULT_METHOD = ImpliedMethod.LOGARITHMIC


@dataclass(frozen=True)
class FairOdds:
    """De-vigged probabilities and margin for a market."""
    probabilities: dict[str, float]
    margin: float
    method: ImpliedMethod

    def __getitem__(self, key: str) -> float:
        return self.probabilities[key]


def implied_probabilities(
    odds: Sequence[float] | Sequence[str],
    market_names: Sequence[str] = ("home", "draw", "away"),
    *,
    method: ImpliedMethod = DEFAULT_METHOD,
    odds_format: OddsFormat = OddsFormat.DECIMAL,
) -> FairOdds:
    """Return de-vigged probabilities and margin via penaltyblog's engine.

    >>> implied_probabilities([2.10, 3.40, 3.60]).probabilities["home"]  # doctest: +SKIP
    0.4487...
    """
    result: ImpliedProbabilities = calculate_implied(
        list(odds),
        method=method,
        odds_format=odds_format,
        market_names=list(market_names),
    )
    return FairOdds(
        probabilities={name: float(result[name]) for name in market_names},
        margin=float(result.margin),
        method=method,
    )


def no_vig_probability(
    decimal_odds: float,
    all_decimal_odds: Sequence[float],
    *,
    method: ImpliedMethod = DEFAULT_METHOD,
) -> tuple[float, float]:
    """Return (fair_probability, margin) for one outcome in a multi-outcome market.

    Drop-in replacement for the bespoke ``market_metrics`` helper in
    ``best_bets.py`` — uses the same LOGARITHMIC de-vig by default.
    """
    fair = implied_probabilities(list(all_decimal_odds), method=method)
    # Map each odds value to its fair probability by position
    odds_to_prob = {
        o: fair.probabilities[name]
        for o, name in zip(all_decimal_odds, fair.probabilities)
    }
    return odds_to_prob[decimal_odds], fair.margin
