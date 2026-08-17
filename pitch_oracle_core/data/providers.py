"""Provider capability and run-health contracts."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Callable, Generic, TypeVar

Payload = TypeVar("Payload")


class CapabilityStatus(StrEnum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class CapabilityReport:
    name: str
    status: CapabilityStatus
    source: str
    observed_at: datetime | None
    coverage: float
    message: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.coverage <= 1:
            raise ValueError("coverage must be in [0, 1]")
        if self.observed_at is not None and self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")

    @property
    def usable(self) -> bool:
        return self.status in {CapabilityStatus.AVAILABLE, CapabilityStatus.DEGRADED}

    def freshness_minutes(self, now: datetime | None = None) -> float | None:
        if self.observed_at is None:
            return None
        now = now or datetime.now(timezone.utc)
        return (now - self.observed_at).total_seconds() / 60


@dataclass(frozen=True)
class ProviderRun:
    run_id: str
    provider: str
    started_at: datetime
    finished_at: datetime
    rows_read: int
    rows_written: int
    status: str
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("provider run timestamps must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("provider run finished before it started")
        if min(self.rows_read, self.rows_written) < 0:
            raise ValueError("provider row counts cannot be negative")


@dataclass(frozen=True)
class OptionalProviderResult(Generic[Payload]):
    payload: Payload
    capability: CapabilityReport


def load_optional_provider(
    name: str,
    source: str,
    loader: Callable[[], Payload],
    fallback: Callable[[], Payload],
    *,
    coverage: Callable[[Payload], float] | None = None,
    observed_at: datetime | None = None,
    maximum_age_seconds: float | None = None,
    now: datetime | None = None,
) -> OptionalProviderResult[Payload]:
    """Contain optional-provider failure without invalidating the base forecast."""
    timestamp = observed_at or datetime.now(timezone.utc)
    current = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or current.tzinfo is None:
        raise ValueError("provider timestamps must be timezone-aware")
    if maximum_age_seconds is not None and maximum_age_seconds <= 0:
        raise ValueError("maximum_age_seconds must be positive")
    try:
        payload = loader()
        observed_coverage = float(coverage(payload) if coverage else 1.0)
    except Exception as exc:  # provider SDKs raise heterogeneous exceptions
        payload = fallback()
        report = CapabilityReport(
            name=name, status=CapabilityStatus.FAILED, source=source,
            observed_at=timestamp, coverage=0.0,
            message=f"{type(exc).__name__}: {exc}; base forecast remains available",
        )
    else:
        stale = (
            maximum_age_seconds is not None
            and (current - timestamp).total_seconds() > maximum_age_seconds
        )
        if stale:
            payload = fallback()
            status = CapabilityStatus.STALE
            message = "Observation exceeded freshness SLO; base forecast remains available"
        else:
            status = (
                CapabilityStatus.AVAILABLE if observed_coverage >= 0.999
                else CapabilityStatus.DEGRADED
            )
            message = ""
        report = CapabilityReport(
            name=name, status=status, source=source, observed_at=timestamp,
            coverage=observed_coverage,
            message=message,
        )
    return OptionalProviderResult(payload, report)
