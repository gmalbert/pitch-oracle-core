"""Validity-interval snapshots with strict pre-kickoff observation boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar

Payload = TypeVar("Payload")


@dataclass(frozen=True)
class Snapshot(Generic[Payload]):
    entity_id: str
    observed_at: datetime
    effective_from: datetime
    effective_to: datetime | None
    source: str
    source_event_id: str
    payload: Payload

    def __post_init__(self) -> None:
        values = [self.observed_at, self.effective_from]
        if self.effective_to is not None:
            values.append(self.effective_to)
        if any(value.tzinfo is None for value in values):
            raise ValueError("snapshot timestamps must be timezone-aware")
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("snapshot effective interval is empty or reversed")

    def usable_for(self, kickoff_utc: datetime) -> bool:
        if kickoff_utc.tzinfo is None:
            raise ValueError("kickoff must be timezone-aware")
        return (
            self.observed_at < kickoff_utc
            and self.effective_from <= kickoff_utc
            and (self.effective_to is None or kickoff_utc < self.effective_to)
        )


def latest_usable_snapshot(snapshots, kickoff_utc):
    usable = [item for item in snapshots if item.usable_for(kickoff_utc)]
    return max(usable, key=lambda item: item.observed_at) if usable else None
