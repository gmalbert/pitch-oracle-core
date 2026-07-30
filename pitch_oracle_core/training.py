"""Stable training entrypoint wrapping the migrated model pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import LeagueConfig
from .runtime import Runtime


@dataclass(frozen=True)
class TrainingResult:
    league: str
    data_path: Path
    models_dir: Path
    metrics: dict[str, Any]


def train(league: LeagueConfig, *, root: str = ".", **kwargs: Any) -> TrainingResult:
    """Train the existing ensemble pipeline under a league-specific runtime.

    The legacy implementation is imported lazily so consumers can import the core
    package without requiring every optional ML dependency at startup.
    """
    runtime = Runtime.for_league(league, root).apply()
    from train_models import train_and_save_models
    metrics = train_and_save_models(**kwargs) or {}
    return TrainingResult(league.key, runtime.data_dir, runtime.models_dir, metrics)


def evaluate(league: LeagueConfig, *, root: str = ".", filename: str = "combined_historical_data_with_calculations.csv") -> dict[str, Any]:
    runtime = Runtime.for_league(league, root)
    from models.poisson_evaluation import evaluate_poisson_file
    return evaluate_poisson_file(str(runtime.data_dir / filename))

