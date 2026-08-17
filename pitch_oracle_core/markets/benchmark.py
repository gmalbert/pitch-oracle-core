"""Market baselines, consensus, and closing-line audit primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import log
import numpy as np

from .devig import DevigMethod, FairMarket, devig
from pitch_oracle_core.domain.forecasts import markets_from_score_matrix
from pitch_oracle_core.models.independent_poisson import independent_poisson_grid


@dataclass(frozen=True)
class OddsSnapshot:
    fixture_id: str
    market: str
    bookmaker: str
    observed_at: datetime
    decimal_odds: tuple[float, ...]
    source_event_id: str
    executable: bool = False

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ValueError("odds observation must be timezone-aware")
        if len(self.decimal_odds) < 2 or any(price <= 1 for price in self.decimal_odds):
            raise ValueError("a market requires at least two valid decimal prices")


def fair_snapshot(snapshot: OddsSnapshot, method: DevigMethod) -> FairMarket:
    return devig(np.asarray(snapshot.decimal_odds, dtype=float), method)


def log_closing_line_value(taken_decimal: float, closing_decimal: float) -> float:
    if taken_decimal <= 1 or closing_decimal <= 1:
        raise ValueError("decimal prices must exceed one")
    return log(taken_decimal / closing_decimal)


def consensus_probability(
    markets: list[FairMarket], *, weights: np.ndarray | None = None
) -> np.ndarray:
    if not markets:
        raise ValueError("at least one fair market is required")
    values = np.vstack([market.probabilities for market in markets])
    if len({len(row) for row in values}) != 1:
        raise ValueError("market outcome dimensions disagree")
    if weights is None:
        weights = np.ones(len(values), dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(values),) or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("invalid consensus weights")
    result = np.average(values, axis=0, weights=weights)
    return result / result.sum()


@dataclass(frozen=True)
class MarketImpliedGoals:
    fixture_id: str
    issued_at: datetime
    expected_home: float
    expected_away: float
    source_markets: tuple[str, ...]
    devig_method: str
    solver_error: float


@dataclass(frozen=True)
class ClosingLineAudit:
    fixture_id: str
    bookmaker: str
    market: str
    selection: str
    accepted_observed_at: datetime
    closing_observed_at: datetime
    accepted_price: float
    closing_price: float
    clv: float
    devig_method: str
    source_coverage: float


def infer_market_implied_goals(
    *,
    fixture_id: str,
    issued_at: datetime,
    one_x_two: np.ndarray,
    over_2_5: float | None = None,
    expected_goal_difference: float | None = None,
    source_markets: tuple[str, ...] = ("1x2",),
    devig_method: str = "multiplicative",
) -> MarketImpliedGoals:
    """Fit an auditable Poisson comparator and retain inconsistency as residual error."""
    if issued_at.tzinfo is None:
        raise ValueError("issued_at must be timezone-aware")
    target = np.asarray(one_x_two, dtype=float)
    if target.shape != (3,) or (target <= 0).any() or not np.isclose(target.sum(), 1.0):
        raise ValueError("1X2 fair probabilities must be positive and sum to one")
    if over_2_5 is not None and not 0 < over_2_5 < 1:
        raise ValueError("over_2_5 must be strictly between zero and one")
    if expected_goal_difference is not None and not np.isfinite(expected_goal_difference):
        raise ValueError("expected goal difference must be finite")

    def residual(home_rate: float, away_rate: float) -> float:
        grid = independent_poisson_grid(home_rate, away_rate)
        matrix = grid.normalized_mass()
        derived = markets_from_score_matrix(matrix)
        values = [
            float(derived["p_home"]) - target[0],
            float(derived["p_draw"]) - target[1],
            float(derived["p_away"]) - target[2],
        ]
        if over_2_5 is not None:
            values.append(float(derived["p_over_2_5"]) - over_2_5)
        if expected_goal_difference is not None:
            values.append((home_rate - away_rate - expected_goal_difference) / 3.0)
        return float(np.mean(np.square(values)))

    best = (float("inf"), 1.4, 1.1)
    for home_rate in np.arange(0.20, 4.01, 0.10):
        for away_rate in np.arange(0.20, 4.01, 0.10):
            score = residual(float(home_rate), float(away_rate))
            if score < best[0]:
                best = (score, float(home_rate), float(away_rate))
    for step in (0.025, 0.005):
        _, center_home, center_away = best
        for home_rate in np.arange(max(0.05, center_home - 4 * step), center_home + 4.1 * step, step):
            for away_rate in np.arange(max(0.05, center_away - 4 * step), center_away + 4.1 * step, step):
                score = residual(float(home_rate), float(away_rate))
                if score < best[0]:
                    best = (score, float(home_rate), float(away_rate))
    error, home_rate, away_rate = best
    return MarketImpliedGoals(
        fixture_id, issued_at, home_rate, away_rate, source_markets,
        devig_method, float(np.sqrt(error)),
    )


def closing_line_audit(
    accepted: OddsSnapshot,
    closing: OddsSnapshot,
    *,
    selection_index: int,
    devig_method: DevigMethod,
    source_coverage: float,
) -> ClosingLineAudit:
    if not 0 <= source_coverage <= 1:
        raise ValueError("source coverage must be in [0, 1]")
    if (
        accepted.fixture_id != closing.fixture_id
        or accepted.market != closing.market
        or accepted.bookmaker != closing.bookmaker
    ):
        raise ValueError("accepted and closing snapshots must describe one market source")
    if closing.observed_at <= accepted.observed_at:
        raise ValueError("closing observation must follow accepted observation")
    if not 0 <= selection_index < len(accepted.decimal_odds):
        raise IndexError("selection index is outside the market")
    accepted_price = accepted.decimal_odds[selection_index]
    closing_price = closing.decimal_odds[selection_index]
    return ClosingLineAudit(
        accepted.fixture_id, accepted.bookmaker, accepted.market,
        str(selection_index), accepted.observed_at, closing.observed_at,
        accepted_price, closing_price,
        log_closing_line_value(accepted_price, closing_price),
        devig_method.value, float(source_coverage),
    )
