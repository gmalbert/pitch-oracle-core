"""Versioned, provider-independent identifiers and observation semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

INTEGRATION_SCHEMA_VERSION = 2
PROVIDER = "pitchapi"


class LineupStatus(StrEnum):
    PREDICTED = "predicted"
    CONFIRMED = "confirmed"
    UNKNOWN = "unknown"


def utc_timestamp(value: datetime | str, *, field: str = "timestamp") -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class RawObservation:
    payload: Any
    observed_at: datetime
    checked_at: datetime
    endpoint: str
    sha256: str
    from_cache: bool = False
    error_code: str | None = None

    def __post_init__(self) -> None:
        utc_timestamp(self.observed_at)
        utc_timestamp(self.checked_at)
        if self.checked_at < self.observed_at:
            raise ValueError("checked_at cannot predate observed_at")


@dataclass(frozen=True)
class PitchAPIFixture:
    match_id: str
    league_id: str
    kickoff_utc: datetime
    status: str
    home_provider_team_id: str
    away_provider_team_id: str
    home_name: str
    away_name: str
    score_home: int | None = None
    score_away: int | None = None

    def __post_init__(self) -> None:
        utc_timestamp(self.kickoff_utc, field="kickoff_utc")
        if self.home_provider_team_id == self.away_provider_team_id:
            raise ValueError("A fixture cannot contain the same team twice")


@dataclass(frozen=True)
class LineupSnapshotRow:
    fixture_id: str
    provider_match_id: str
    team_id: str
    player_id: str
    player_name: str
    position: str | None
    is_starter: bool
    lineup_status: LineupStatus
    snapshot_at: datetime
    kickoff_utc: datetime

    def __post_init__(self) -> None:
        utc_timestamp(self.snapshot_at, field="snapshot_at")
        utc_timestamp(self.kickoff_utc, field="kickoff_utc")

    @property
    def prematch_eligible(self) -> bool:
        # Post-kickoff responses are kept as observations, but never used to replay a pre-match XI.
        return self.snapshot_at < self.kickoff_utc
