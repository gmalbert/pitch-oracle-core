"""Typed, league-neutral domain contracts used by artifacts and applications."""

from .competitions import (
    BracketRule,
    CompetitionEdition,
    CompetitionRules,
    PhaseRule,
    TieBreaker,
    edition_from_league_config,
    rules_from_league_config,
)
from .entities import (
    EntityResolver,
    Resolution,
    ResolutionStatus,
    Team,
    TeamAlias,
    assert_active_team_coverage,
    normalized_name,
)
from .forecasts import MatchForecast, markets_from_score_matrix
from .probability_grid import ProbabilityGrid
from .catalog import FeatureStatus, feature_catalog_statuses, manifest_feature_statuses
from .research import ResearchInitiative, research_initiatives

__all__ = [
    "CompetitionEdition",
    "CompetitionRules",
    "BracketRule",
    "EntityResolver",
    "MatchForecast",
    "ProbabilityGrid",
    "Resolution",
    "ResolutionStatus",
    "FeatureStatus",
    "ResearchInitiative",
    "Team",
    "TeamAlias",
    "PhaseRule",
    "TieBreaker",
    "assert_active_team_coverage",
    "markets_from_score_matrix",
    "normalized_name",
    "edition_from_league_config",
    "feature_catalog_statuses",
    "manifest_feature_statuses",
    "research_initiatives",
    "rules_from_league_config",
]
