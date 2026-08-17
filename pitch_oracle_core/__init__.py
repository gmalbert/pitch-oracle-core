"""Public API for the Pitch Oracle shared core package."""

from importlib import import_module

from ._version import __version__
from .config import DataSourceConfig, LeagueConfig, PhaseConfig, PlayoffConfig, ThemeConfig
from .leagues import BUILTIN_LEAGUES, get_league_config
from .odds import OddsAdapter, OddsEvent, OddsMarket, normalize_odds
from .phases import (
    apply_phase_transition, assign_phase, build_split_pools,
    eligible_opponents, phase_start_standings,
)
from .sources import OptionalFeatureSet, SourceAvailability
from .runtime import Runtime, current_runtime
from .app_factory import run as run_app
from .providers import ProviderRegistry, Stadium, calculate_shot_xg, require_source
from .providers import PitchAPIXGProvider
from .training import TrainingResult, evaluate, train
from .xg import Shot, expected_goals_from_shots
from .goal_markets import GoalMarketProbabilities, calculate_goal_markets
from .features import (
    FEATURE_POLICY_VERSION,
    chronological_partition_indices,
    chronological_split_indices,
    completed_future_rows,
    completed_match_rows,
    is_market_feature,
    no_odds_feature_columns,
    parse_match_dates,
    prematch_feature_columns,
    prior_group_rolling,
)
from .risk import calculate_prediction_risk, get_prediction_guidance, get_risk_category
from .predictions import (
    FeatureContract,
    build_prediction_frame,
    build_upcoming_feature_matrix,
    production_probabilities,
)

_LAZY_EXPORTS = {
    "add_weather_features": (".weather", "add_weather_features"),
    "add_weather_impact_category": (".weather", "add_weather_impact_category"),
    "fetch_match_weather": (".weather", "fetch_match_weather"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module = import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value

__all__ = [
    "__version__", "BUILTIN_LEAGUES", "DataSourceConfig", "LeagueConfig", "PhaseConfig", "ThemeConfig",
    "OddsAdapter", "OddsEvent", "OddsMarket", "OptionalFeatureSet",
    "PlayoffConfig", "Runtime", "Shot", "SourceAvailability", "apply_phase_transition",
    "assign_phase", "build_split_pools", "eligible_opponents",
    "phase_start_standings", "expected_goals_from_shots",
    "ProviderRegistry", "Stadium", "TrainingResult", "calculate_shot_xg",
    "PitchAPIXGProvider",
    "current_runtime", "evaluate", "get_league_config", "normalize_odds",
    "require_source", "run_app", "train", "GoalMarketProbabilities",
    "calculate_goal_markets", "FEATURE_POLICY_VERSION", "chronological_partition_indices",
    "chronological_split_indices", "completed_future_rows", "completed_match_rows",
    "is_market_feature", "no_odds_feature_columns", "parse_match_dates",
    "prematch_feature_columns", "prior_group_rolling", "calculate_prediction_risk",
    "get_prediction_guidance", "get_risk_category",
    "FeatureContract", "build_prediction_frame", "build_upcoming_feature_matrix",
    "production_probabilities",
    "add_weather_features", "add_weather_impact_category", "fetch_match_weather",
]
