"""Public API for the Pitch Oracle shared core package."""

from .config import DataSourceConfig, LeagueConfig, PhaseConfig, PlayoffConfig
from .leagues import BUILTIN_LEAGUES, get_league_config
from .odds import OddsAdapter, OddsEvent, OddsMarket, normalize_odds
from .phases import apply_phase_transition, assign_phase, eligible_opponents
from .sources import OptionalFeatureSet, SourceAvailability
from .runtime import Runtime, current_runtime
from .app_factory import run as run_app
from .providers import ProviderRegistry, Stadium, calculate_shot_xg, require_source
from .training import TrainingResult, evaluate, train
from .xg import Shot, expected_goals_from_shots
from .goal_markets import GoalMarketProbabilities, calculate_goal_markets

__all__ = [
    "BUILTIN_LEAGUES", "DataSourceConfig", "LeagueConfig", "PhaseConfig",
    "OddsAdapter", "OddsEvent", "OddsMarket", "OptionalFeatureSet",
    "PlayoffConfig", "Runtime", "Shot", "SourceAvailability", "apply_phase_transition",
    "assign_phase", "eligible_opponents", "expected_goals_from_shots",
    "ProviderRegistry", "Stadium", "TrainingResult", "calculate_shot_xg",
    "current_runtime", "evaluate", "get_league_config", "normalize_odds",
    "require_source", "run_app", "train", "GoalMarketProbabilities",
    "calculate_goal_markets",
]
