"""Optional feature handling for sources that do not cover every league."""

from dataclasses import dataclass
from typing import Any

from .config import DataSourceConfig


@dataclass(frozen=True)
class SourceAvailability:
    referee: bool = False
    injuries: bool = False
    live_odds: bool = False
    pitchapi: bool = False

    @classmethod
    def from_config(cls, config: DataSourceConfig) -> "SourceAvailability":
        return cls(
            config.referee, config.injuries, bool(config.live_odds_providers),
            config.pitchapi,
        )


@dataclass(frozen=True)
class OptionalFeatureSet:
    referee: dict[str, Any] | None = None
    injuries: dict[str, Any] | None = None
    live_odds: dict[str, Any] | None = None

    def as_model_features(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in (("referee", self.referee), ("injuries", self.injuries), ("live_odds", self.live_odds)):
            if value is not None:
                result.update({f"{name}_{key}": item for key, item in value.items()})
        return result

