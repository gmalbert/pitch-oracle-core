"""Provider-neutral live odds contract. Providers are league-specific adapters."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class OddsMarket:
    outcome: str
    decimal_price: float
    bookmaker: str | None = None


@dataclass(frozen=True)
class OddsEvent:
    event_id: str
    home_team: str
    away_team: str
    start_time: str | None
    markets: tuple[OddsMarket, ...]
    provider: str


class OddsAdapter(Protocol):
    name: str
    def fetch(self, *, league: str, date: str | None = None) -> list[OddsEvent]: ...


def normalize_odds(payload: dict[str, Any], *, provider: str) -> OddsEvent:
    """Normalize a common provider payload and fail loudly on malformed prices."""
    markets = tuple(
        OddsMarket(str(item["outcome"]), float(item["decimal_price"]), item.get("bookmaker"))
        for item in payload.get("markets", [])
    )
    if any(m.decimal_price <= 1 for m in markets):
        raise ValueError("decimal odds must be greater than 1")
    return OddsEvent(str(payload["event_id"]), str(payload["home_team"]), str(payload["away_team"]),
                     payload.get("start_time"), markets, provider)

