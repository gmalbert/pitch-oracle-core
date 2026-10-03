"""Capability-separated PitchAPI ingestion with immutable source lineage."""

from .client import PitchAPIClient, PitchAPIError
from .contracts import INTEGRATION_SCHEMA_VERSION, LineupStatus

__all__ = ["PitchAPIClient", "PitchAPIError", "INTEGRATION_SCHEMA_VERSION", "LineupStatus"]
