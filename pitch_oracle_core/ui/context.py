"""Shared application context passed to every page renderer."""

from dataclasses import dataclass

from pitch_oracle_core.config import LeagueConfig
from .repository import ArtifactRepository


@dataclass(frozen=True)
class AppContext:
    config: LeagueConfig
    repository: ArtifactRepository
    capabilities: dict[str, dict]
    edition_id: str
    scenario_adapter: object | None = None
    display_timezone: str = "UTC"

    def has_capability(self, name: str) -> bool:
        return self.capabilities.get(name, {}).get("status") in {
            "available", "degraded", "partial"
        }

    def capability_message(self, name: str) -> str:
        report = self.capabilities.get(name, {})
        return str(report.get("message") or report.get("reason") or "Not available")
