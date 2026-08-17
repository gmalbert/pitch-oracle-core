"""Stable provider contracts used by thin league repositories."""

from dataclasses import dataclass
import os
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
class PitchAPIXGProvider:
    """Match-level xG from PitchAPI, keyed for the ledger merge.

    Columns: match_id, match_date, home_team_id, away_team_id, home_xg, away_xg.
    Uses the configured ``pitchapi_league_id`` and the ``PITCH_API_KEY`` env var.
    """

    api_key: str | None = None

    def fetch(self, league: LeagueConfig, season: str) -> pd.DataFrame:
        from ..fetch_pitchapi import PitchAPIClient, pitchapi_league_id

        league_id = pitchapi_league_id(league)
        key = self.api_key or os.getenv("PITCH_API_KEY")
        if not key:
            raise RuntimeError("PITCH_API_KEY is required to fetch PitchAPI xG")
        client = PitchAPIClient(api_key=key)
        matches = client.league_matches(league_id, season)
        rows = []
        for match in matches:
            if match.get("status") != "finished":
                continue
            periods = client.match_shots(match["id"])
            home_xg = sum(
                float(shot.get("expected_goals") or 0.0)
                for period in periods
                for shot in period.get("shots", [])
                if shot.get("team_id") == match.get("home_team", {}).get("id")
            )
            away_xg = sum(
                float(shot.get("expected_goals") or 0.0)
                for period in periods
                for shot in period.get("shots", [])
                if shot.get("team_id") == match.get("away_team", {}).get("id")
            )
            rows.append({
                "match_id": match["id"],
                "match_date": match.get("date"),
                "home_team_id": match.get("home_team", {}).get("id"),
                "away_team_id": match.get("away_team", {}).get("id"),
                "home_xg": round(home_xg, 4),
                "away_xg": round(away_xg, 4),
            })
        return pd.DataFrame(rows)


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

