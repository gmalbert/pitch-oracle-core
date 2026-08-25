"""Betting utilities backed by penaltyblog.betting — Kelly, value, arbitrage, hedge.

Provides the full betting toolkit: Kelly criterion sizing, value-bet
detection, arbitrage scanning, and in-play hedge computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from penaltyblog.betting import (
    arbitrage_hedge,
    find_arbitrage_opportunities,
    identify_value_bet,
    kelly_criterion,
    multiple_kelly_criterion,
)
from penaltyblog.implied import ImpliedMethod

from .implied import implied_probabilities


@dataclass(frozen=True)
class ValueBetResult:
    """A single value-bet opportunity."""
    outcome: str
    decimal_odds: float
    model_probability: float
    market_probability: float
    edge: float
    expected_value: float
    kelly_stake: float
    tier: str


@dataclass(frozen=True)
class ArbitrageResult:
    """An arbitrage opportunity across bookmakers."""
    has_arbitrage: bool
    guaranteed_return: float
    best_odds: list[float]
    best_bookmakers: list[str]
    stake_percentages: list[float]


def compute_kelly_stake(
    decimal_odds: float,
    true_prob: float,
    fraction: float = 0.5,
) -> dict[str, float]:
    """Compute Kelly criterion stake for a single bet."""
    result = kelly_criterion(decimal_odds=decimal_odds, true_prob=true_prob, fraction=fraction)
    return {
        "stake": result.stake,
        "expected_growth": result.expected_growth,
        "edge": result.edge,
        "is_favorable": result.is_favorable,
    }


def compute_multi_kelly(
    decimal_odds: list[float],
    true_probs: list[float],
    fraction: float = 0.5,
) -> dict[str, float]:
    """Compute multi-bet Kelly stakes for a portfolio of independent bets."""
    result = multiple_kelly_criterion(
        decimal_odds=decimal_odds,
        true_probs=true_probs,
        fraction=fraction,
    )
    return {
        "total_stake": result.total_stake,
        "portfolio_edge": result.portfolio_edge,
    }


def find_value_bets(
    model_probabilities: list[float],
    decimal_odds: list[float],
    market_names: list[str] | None = None,
    kelly_fraction: float = 0.5,
) -> list[ValueBetResult]:
    """Identify value bets by comparing model probabilities to market odds."""
    if market_names is None:
        market_names = ["home", "draw", "away"]
    result = identify_value_bet(
        bookmaker_odds=decimal_odds,
        estimated_probability=model_probabilities,
        kelly_fraction=kelly_fraction,
    )
    bets = []
    for i, vr in enumerate(result.individual_results):
        if vr.is_value_bet:
            fair = implied_probabilities(decimal_odds)
            bets.append(ValueBetResult(
                outcome=market_names[i],
                decimal_odds=decimal_odds[i],
                model_probability=model_probabilities[i],
                market_probability=fair[market_names[i]],
                edge=vr.edge,
                expected_value=vr.expected_value,
                kelly_stake=vr.recommended_stake_fraction,
                tier=_tier(vr.expected_value),
            ))
    return bets


def find_arbitrage(
    bookmaker_odds: list[list[float]],
    outcome_labels: list[str] | None = None,
) -> ArbitrageResult:
    """Scan for arbitrage opportunities across bookmakers."""
    if outcome_labels is None:
        outcome_labels = ["home", "draw", "away"]
    arb = find_arbitrage_opportunities(bookmaker_odds, outcome_labels=outcome_labels)
    return ArbitrageResult(
        has_arbitrage=arb.has_arbitrage,
        guaranteed_return=arb.guaranteed_return if arb.has_arbitrage else 0.0,
        best_odds=list(arb.best_odds) if arb.has_arbitrage else [],
        best_bookmakers=list(arb.best_bookmakers) if arb.has_arbitrage else [],
        stake_percentages=list(arb.stake_percentages) if arb.has_arbitrage else [],
    )


def compute_hedge(
    existing_stakes: list[float],
    existing_odds: list[float],
    hedge_odds: list[float],
    target_profit: float = 0.0,
) -> dict[str, float]:
    """Compute hedge stakes for existing positions."""
    result = arbitrage_hedge(
        existing_stakes=existing_stakes,
        existing_odds=existing_odds,
        hedge_odds=hedge_odds,
        target_profit=target_profit,
    )
    return {
        "hedge_stakes": list(result.practical_hedge_stakes),
        "guaranteed_profit": result.guaranteed_profit,
    }


def _tier(expected_value: float) -> str:
    if expected_value >= 0.12:
        return "Elite"
    if expected_value >= 0.07:
        return "Strong"
    if expected_value >= 0.03:
        return "Good"
    return "Standard"
