"""Allowlisted point-in-time feature definitions with lineage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable
import pandas as pd


class FeatureFamily(StrEnum):
    FORM = "form"
    STRENGTH = "strength"
    SCHEDULE = "schedule"
    SQUAD = "squad"
    WEATHER = "weather"
    MARKET = "market"


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    family: FeatureFamily
    dtype: str
    builder: Callable[[pd.DataFrame], pd.Series]
    max_age_hours: int | None = None
    optional_capability: str | None = None
    scenario_mutable: bool = False


class FeatureRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, FeatureDefinition] = {}

    def register(self, definition: FeatureDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Duplicate feature {definition.name!r}")
        self._definitions[definition.name] = definition

    def build(
        self,
        source: pd.DataFrame,
        *,
        kickoff_column: str = "kickoff_utc",
        observed_column: str = "observed_at",
    ) -> pd.DataFrame:
        required = {kickoff_column, observed_column, "fixture_id"}
        missing = required.difference(source.columns)
        if missing:
            raise ValueError(f"Missing point-in-time columns: {sorted(missing)}")
        kickoff = pd.to_datetime(source[kickoff_column], utc=True, errors="coerce")
        observed = pd.to_datetime(source[observed_column], utc=True, errors="coerce")
        invalid = kickoff.isna() | observed.isna()
        if invalid.any():
            raise ValueError("Every feature row needs valid kickoff and observed timestamps")
        leakage = observed >= kickoff
        if leakage.any():
            fixtures = source.loc[leakage, "fixture_id"].head(5).tolist()
            raise ValueError(f"Feature observations are not pre-kickoff: {fixtures}")
        output: dict[str, pd.Series] = {}
        for name, definition in self._definitions.items():
            age_hours = (kickoff - observed).dt.total_seconds() / 3600
            values = definition.builder(source)
            if definition.max_age_hours is not None:
                values = values.mask(age_hours > definition.max_age_hours)
            output[name] = values.astype(definition.dtype)
        return pd.DataFrame(output, index=source.index)

    def metadata(self) -> list[dict[str, object]]:
        return [
            {
                "name": item.name,
                "family": item.family.value,
                "dtype": item.dtype,
                "max_age_hours": item.max_age_hours,
                "optional_capability": item.optional_capability,
                "scenario_mutable": item.scenario_mutable,
            }
            for item in self._definitions.values()
        ]

    def scenario_mutable_names(self) -> frozenset[str]:
        return frozenset(
            item.name for item in self._definitions.values() if item.scenario_mutable
        )
