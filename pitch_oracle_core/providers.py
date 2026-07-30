"""Stable provider contracts used by thin league repositories."""

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from .config import LeagueConfig
from .xg import Shot, expected_goals_from_shots


class HistoricalProvider(Protocol):
    def fetch(self, league: LeagueConfig, seasons: list[str]) -> pd.DataFrame: ...


class FixtureProvider(Protocol):
    def fetch(self, league: LeagueConfig, days_ahead: int = 60) -> pd.DataFrame: ...


class OptionalFeatureProvider(Protocol):
    name: str
    def fetch(self, league: LeagueConfig, **kwargs: Any) -> dict[str, Any] | None: ...


class XGProvider(Protocol):
    def fetch(self, league: LeagueConfig, season: str) -> pd.DataFrame: ...


@dataclass(frozen=True)
class Stadium:
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class ProviderRegistry:
    historical: HistoricalProvider | None = None
    fixtures: FixtureProvider | None = None
    xg: XGProvider | None = None
    referee: OptionalFeatureProvider | None = None
    injuries: OptionalFeatureProvider | None = None
    odds: Any | None = None

    def fetch_optional_features(self, league: LeagueConfig, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for provider in (self.referee, self.injuries):
            if provider is None or not getattr(league.sources, provider.name, False):
                continue
            value = provider.fetch(league, **kwargs)
            if value:
                result[provider.name] = value
        return result


def calculate_shot_xg(shots: list[Shot]) -> float:
    """Canonical xG calculation for all non-Understat leagues."""
    return expected_goals_from_shots(shots)


def require_source(league: LeagueConfig, source: str) -> None:
    if not getattr(league.sources, source, False):
        raise RuntimeError(f"{source} is disabled or unavailable for {league.display_name}")

