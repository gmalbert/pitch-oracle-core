"""Canonical action coordinates and provider lineage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Mapping


class ActionType(StrEnum):
    PASS = "pass"
    CARRY = "carry"
    DRIBBLE = "dribble"
    SHOT = "shot"
    DUEL = "duel"
    RECOVERY = "recovery"
    FOUL = "foul"
    KEEPER = "keeper"
    OTHER = "other"


@dataclass(frozen=True)
class CanonicalAction:
    fixture_id: str
    action_id: str
    sequence_index: int
    period: int
    seconds_in_period: float
    team_id: str
    player_id: str | None
    action_type: ActionType
    start_x: float | None
    start_y: float | None
    end_x: float | None
    end_y: float | None
    successful: bool | None
    body_part: str | None
    set_piece: str | None
    observed_at: datetime
    provider: str
    provider_event_id: str
    provider_schema_version: str
    raw_attributes: Mapping[str, object]

    def validate(self) -> None:
        if self.sequence_index < 0 or self.period < 1 or self.seconds_in_period < 0:
            raise ValueError("invalid event ordering fields")
        if self.observed_at.tzinfo is None:
            raise ValueError("event observation must be timezone-aware")
        for x in (self.start_x, self.end_x):
            if x is not None and not 0 <= x <= 105:
                raise ValueError("x coordinate outside canonical pitch")
        for y in (self.start_y, self.end_y):
            if y is not None and not 0 <= y <= 68:
                raise ValueError("y coordinate outside canonical pitch")


class CapabilityStatus(StrEnum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class ProviderCapability:
    competition_id: str
    edition_id: str
    capability: str
    status: CapabilityStatus
    coverage_fraction: float
    observed_at: datetime
    provider: str
    reason: str | None

    def __post_init__(self) -> None:
        if not 0 <= self.coverage_fraction <= 1:
            raise ValueError("coverage_fraction must be in [0, 1]")
        if self.observed_at.tzinfo is None:
            raise ValueError("capability observation must be timezone-aware")
