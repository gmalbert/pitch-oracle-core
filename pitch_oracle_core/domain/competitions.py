"""Competition-edition identity, local-time display, and season rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from zoneinfo import ZoneInfo


class TieBreaker(StrEnum):
    POINTS = "points"
    GOAL_DIFFERENCE = "goal_difference"
    GOALS_FOR = "goals_for"
    HEAD_TO_HEAD_POINTS = "head_to_head_points"
    WINS = "wins"


@dataclass(frozen=True)
class BracketRule:
    bracket_id: str
    sources: tuple[str, ...]
    legs: int = 1
    outcome_label: str | None = None

    def __post_init__(self) -> None:
        if not self.bracket_id or self.legs < 1:
            raise ValueError("bracket ID and positive legs are required")


@dataclass(frozen=True)
class PhaseRule:
    phase_id: str
    starts_after_round: int | None = None
    pool_sizes: tuple[int, ...] = ()
    pool_labels: tuple[str, ...] = ()
    points_multiplier: float = 1.0
    points_rounding: str = "none"
    fixture_repeats: int = 1
    brackets: tuple[BracketRule, ...] = ()

    def __post_init__(self) -> None:
        if self.pool_labels and len(self.pool_labels) != len(self.pool_sizes):
            raise ValueError("phase pool labels and sizes disagree")
        if self.points_multiplier <= 0:
            raise ValueError("points_multiplier must be positive")
        if self.fixture_repeats < 1:
            raise ValueError("fixture_repeats must be positive")
        if self.points_rounding not in {"none", "ceil", "floor"}:
            raise ValueError("unsupported points rounding")


@dataclass(frozen=True)
class CompetitionRules:
    version: str
    win_points: int = 3
    draw_points: int = 1
    tie_breakers: tuple[TieBreaker, ...] = (
        TieBreaker.POINTS,
        TieBreaker.GOAL_DIFFERENCE,
        TieBreaker.GOALS_FOR,
    )
    phases: tuple[PhaseRule, ...] = (PhaseRule("regular"),)
    points_adjustments: dict[str, int] = field(default_factory=dict)
    outcome_labels: dict[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class CompetitionEdition:
    edition_id: str
    competition_id: str
    display_name: str
    timezone: str
    season_start_month: int
    team_ids: tuple[str, ...]
    rules_version: str

    def __post_init__(self) -> None:
        if not 1 <= self.season_start_month <= 12:
            raise ValueError("season_start_month must be between 1 and 12")
        ZoneInfo(self.timezone)

    def season_id(self, kickoff_utc: datetime) -> str:
        if kickoff_utc.tzinfo is None:
            raise ValueError("kickoff_utc must be timezone-aware")
        local = kickoff_utc.astimezone(ZoneInfo(self.timezone))
        start = local.year if local.month >= self.season_start_month else local.year - 1
        return f"{start}-{str(start + 1)[-2:]}"


def parse_provider_kickoff(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Provider kickoff must include an offset")
    return parsed.astimezone(ZoneInfo("UTC"))


def kickoff_for_display(kickoff_utc: datetime, timezone: str) -> datetime:
    if kickoff_utc.tzinfo is None:
        raise ValueError("kickoff_utc must be timezone-aware")
    return kickoff_utc.astimezone(ZoneInfo(timezone))


def edition_from_league_config(
    config,
    season_start_year: int,
    team_ids: tuple[str, ...] = (),
) -> CompetitionEdition:
    """Adapt the legacy LeagueConfig into an explicit competition edition."""
    if season_start_year < 1900:
        raise ValueError("invalid season start year")
    season_id = f"{season_start_year}-{str(season_start_year + 1)[-2:]}"
    competition_id = config.espn_slug or config.key
    rules_version = f"{competition_id}-{season_id}-v1"
    return CompetitionEdition(
        edition_id=f"{competition_id}:{season_id}",
        competition_id=competition_id,
        display_name=config.display_name,
        timezone=config.sources.weather_timezone,
        season_start_month=config.season_months[0],
        team_ids=team_ids,
        rules_version=rules_version,
    )


def rules_from_league_config(config, *, version: str | None = None) -> CompetitionRules:
    """Convert data-driven legacy phase configuration into versioned rules."""
    phases = [PhaseRule("regular")]
    if config.phase.split_after_round is not None:
        phases.append(PhaseRule(
            "split",
            starts_after_round=config.phase.split_after_round,
            pool_sizes=config.phase.split_pool_sizes,
            pool_labels=config.phase.split_pools,
            points_multiplier=0.5 if config.phase.points_halving else 1.0,
            points_rounding=config.phase.points_halving_rounding,
            fixture_repeats=1,
        ))
    for playoff in config.phase.playoffs:
        phases.append(PhaseRule(
            playoff.name,
            brackets=(BracketRule(
                bracket_id=playoff.name,
                sources=playoff.sources,
                legs=playoff.legs,
                outcome_label=playoff.outcome_label,
            ),),
        ))
    return CompetitionRules(
        version=version or f"{config.key}-rules-v1",
        tie_breakers=tuple(TieBreaker(item) for item in config.tie_breakers),
        phases=tuple(phases),
        points_adjustments=dict(config.points_adjustments),
        outcome_labels=dict(config.outcome_labels),
    )
