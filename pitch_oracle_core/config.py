"""Configuration contracts. League behavior belongs in data, not conditionals."""

from dataclasses import dataclass, field
from typing import Literal

PhaseKind = Literal["regular", "split", "playoff"]


@dataclass(frozen=True)
class DataSourceConfig:
    historical_results: bool = True
    historical_odds: bool = True
    espn_slug: str | None = None
    clubelo_code: str | None = None
    api_football: bool = True
    understat: bool = False
    weather: bool = True
    referee: bool = False
    injuries: bool = False
    live_odds_providers: tuple[str, ...] = ()
    api_football_league_id: int | None = None
    understat_league: str | None = None
    referee_endpoint: str | None = None
    injury_endpoint: str | None = None


@dataclass(frozen=True)
class ThemeConfig:
    """Consumer-level visual overrides for the shared core theme."""

    primary: str = "#1554a6"
    primary_dark: str = "#0d2f5f"
    sidebar: str = "#f1f4f8"
    page: str = "#ffffff"
    border: str = "#d9e0e8"
    muted: str = "#64748b"


@dataclass(frozen=True)
class PlayoffConfig:
    name: str
    phase_kind: PhaseKind = "playoff"
    pools: tuple[str, ...] = ()
    cross_division: bool = False
    included_in_v1: bool = True


@dataclass(frozen=True)
class PhaseConfig:
    regular_matches_per_opponent: int = 2
    split_after_round: int | None = None
    split_pools: tuple[str, ...] = ()
    points_halving: bool = False
    points_halving_rounding: Literal["ceil", "floor", "none"] = "ceil"
    playoffs: tuple[PlayoffConfig, ...] = ()


@dataclass(frozen=True)
class LeagueConfig:
    key: str
    display_name: str
    football_data_div: str
    espn_slug: str | None
    clubelo_code: str | None
    team_count: int
    season_months: tuple[int, int]
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    points_adjustments: dict[str, int] = field(default_factory=dict)
    stadium_coordinates: dict[str, tuple[float, float]] = field(default_factory=dict)
    team_aliases: dict[str, str] = field(default_factory=dict)
    sources: DataSourceConfig = field(default_factory=DataSourceConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    data_dir_name: str = "data_files"
    models_dir_name: str = "models"
