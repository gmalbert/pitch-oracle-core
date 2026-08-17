"""Timestamped quote ledgers and robust bookmaker consensus."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuoteStatus:
    observed_at: pd.Timestamp
    age_seconds: float
    fresh: bool
    executable: bool


@dataclass(frozen=True)
class OddsQuote:
    fixture_id: str
    market: str
    selection: str
    decimal_price: float
    bookmaker: str
    observed_at: datetime
    source: str
    executable: bool = False

    def __post_init__(self) -> None:
        if self.decimal_price <= 1 or not np.isfinite(self.decimal_price):
            raise ValueError("Decimal price must exceed 1")
        if self.observed_at.tzinfo is None:
            raise ValueError("Odds observation must be timezone-aware")


def multiplicative_devig(decimal_prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(decimal_prices, dtype=float)
    if prices.ndim != 1 or len(prices) < 2 or (prices <= 1).any():
        raise ValueError("A market needs at least two valid decimal prices")
    implied = 1.0 / prices
    return implied / implied.sum()


def bookmaker_market_probabilities(quotes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["fixture_id", "market", "bookmaker", "observed_at"]
    for key, group in quotes.groupby(keys, observed=True):
        group = group.sort_values("selection")
        fair = multiplicative_devig(group.decimal_price.to_numpy())
        for row, probability in zip(group.itertuples(), fair):
            rows.append({
                **dict(zip(keys, key)),
                "selection": row.selection,
                "decimal_price": row.decimal_price,
                "fair_probability": float(probability),
                "executable": bool(getattr(row, "executable", False)),
            })
    return pd.DataFrame(rows)


def consensus_market(bookmaker_probabilities: pd.DataFrame) -> pd.DataFrame:
    return bookmaker_probabilities.groupby(
        ["fixture_id", "market", "selection"], as_index=False
    ).agg(
        consensus_probability=("fair_probability", "median"),
        market_low=("fair_probability", "min"),
        market_high=("fair_probability", "max"),
        books=("bookmaker", "nunique"),
        best_price=("decimal_price", "max"),
        observed_at=("observed_at", "max"),
    )


def market_movement(bookmaker_probabilities: pd.DataFrame) -> pd.DataFrame:
    """Publish opener/current consensus moves and current source dispersion."""
    required = {
        "fixture_id", "market", "selection", "bookmaker", "observed_at",
        "fair_probability", "decimal_price",
    }
    missing = required.difference(bookmaker_probabilities.columns)
    if missing:
        raise ValueError(f"Market movement misses: {sorted(missing)}")
    frame = bookmaker_probabilities.copy()
    frame["observed_at"] = pd.to_datetime(frame.observed_at, utc=True, errors="coerce")
    if frame.observed_at.isna().any():
        raise ValueError("market movement requires valid observation times")
    rows = []
    keys = ["fixture_id", "market", "selection"]
    for key, group in frame.groupby(keys, observed=True):
        ordered = group.sort_values("observed_at")
        opening_time, current_time = ordered.observed_at.min(), ordered.observed_at.max()
        opening = ordered.loc[ordered.observed_at == opening_time]
        current = ordered.loc[ordered.observed_at == current_time]
        opening_probability = float(opening.fair_probability.median())
        current_probability = float(current.fair_probability.median())
        rows.append({
            **dict(zip(keys, key)),
            "opening_probability": opening_probability,
            "current_probability": current_probability,
            "probability_move": current_probability - opening_probability,
            "opening_observed_at": opening_time,
            "current_observed_at": current_time,
            "current_best_price": float(current.decimal_price.max()),
            "current_dispersion": float(current.fair_probability.max() - current.fair_probability.min()),
            "current_books": int(current.bookmaker.nunique()),
        })
    return pd.DataFrame(rows)


def latest_quotes_before(quotes: pd.DataFrame, issued_at: object) -> pd.DataFrame:
    frame = quotes.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    cutoff = pd.Timestamp(issued_at)
    if cutoff.tzinfo is None:
        raise ValueError("issue time must be timezone-aware")
    available = frame.loc[frame["observed_at"] < cutoff]
    if available.empty:
        return available
    index = available.groupby(
        ["fixture_id", "market", "selection", "bookmaker"], observed=True
    )["observed_at"].idxmax()
    return available.loc[index].reset_index(drop=True)


def quote_status(
    observed_at: object,
    as_of: object,
    *,
    executable: bool,
    maximum_age_seconds: float = 900,
) -> QuoteStatus:
    observed = pd.Timestamp(observed_at)
    current = pd.Timestamp(as_of)
    if observed.tzinfo is None or current.tzinfo is None:
        raise ValueError("quote status timestamps must be timezone-aware")
    age = (current - observed).total_seconds()
    if age < 0:
        raise ValueError("quote was observed after the assessment time")
    if maximum_age_seconds <= 0:
        raise ValueError("maximum quote age must be positive")
    fresh = age <= maximum_age_seconds
    return QuoteStatus(observed, float(age), fresh, bool(executable and fresh))
