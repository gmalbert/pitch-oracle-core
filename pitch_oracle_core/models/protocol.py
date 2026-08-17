"""One protocol and track declaration for every score-model candidate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Mapping, Protocol, Sequence
import pandas as pd

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid


class ForecastTrack(StrEnum):
    INDEPENDENT = "independent"
    MARKET_AWARE = "market_aware"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    track: ForecastTrack
    required_capabilities: frozenset[str]
    hyperparameters: Mapping[str, object]

    def validate(self) -> None:
        if not self.model_id or not self.family:
            raise ValueError("model_id and family are required")
        market_inputs = {
            "odds_1x2", "odds_totals", "odds_handicap", "market_movement"
        }
        if self.track == ForecastTrack.INDEPENDENT:
            forbidden = self.required_capabilities.intersection(market_inputs)
            if forbidden:
                raise ValueError(f"independent model requests market inputs: {forbidden}")


@dataclass(frozen=True)
class FixtureFeatures:
    fixture_id: str
    kickoff_utc: datetime
    home_team_id: str
    away_team_id: str
    values: Mapping[str, float | int | str | None]

    def __post_init__(self) -> None:
        if self.kickoff_utc.tzinfo is None:
            raise ValueError("fixture kickoff must be timezone-aware")


class ScoreModel(Protocol):
    spec: ModelSpec

    def fit(self, matches: pd.DataFrame, *, cutoff_utc: datetime) -> "ScoreModel": ...

    def predict_grid(self, fixture: FixtureFeatures) -> ProbabilityGrid: ...


def validate_fixture_track(spec: ModelSpec, fixture: FixtureFeatures) -> None:
    """Keep odds-derived columns out of independent-model inference rows."""
    spec.validate()
    if spec.track != ForecastTrack.INDEPENDENT:
        return
    forbidden_tokens = ("odds", "market", "price", "bookmaker", "clv")
    forbidden = sorted(
        name for name in fixture.values
        if any(token in str(name).casefold() for token in forbidden_tokens)
    )
    if forbidden:
        raise ValueError(f"independent fixture contains market-derived features: {forbidden}")


def validate_candidate_specs(specs: Sequence[ModelSpec]) -> None:
    seen: set[str] = set()
    for spec in specs:
        spec.validate()
        if spec.model_id in seen:
            raise ValueError(f"duplicate model_id: {spec.model_id}")
        seen.add(spec.model_id)
