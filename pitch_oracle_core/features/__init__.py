"""Point-in-time feature policy and feature-mart builders.

The public policy predates this package layout and remains in the adjacent
``features.py`` compatibility module. Loading it under a private module name keeps
the old API stable while allowing ``pitch_oracle_core.features.ledger`` and future
builders to be packaged normally.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_policy_name = "pitch_oracle_core._legacy_feature_policy"
_policy = sys.modules.get(_policy_name)
if _policy is None:
    _path = Path(__file__).resolve().parent.parent / "features.py"
    _spec = spec_from_file_location(_policy_name, _path)
    if _spec is None or _spec.loader is None:  # pragma: no cover - installation guard
        raise ImportError(f"Unable to load feature policy from {_path}")
    _policy = module_from_spec(_spec)
    sys.modules[_policy_name] = _policy
    _spec.loader.exec_module(_policy)

FEATURE_POLICY_VERSION = _policy.FEATURE_POLICY_VERSION
chronological_partition_indices = _policy.chronological_partition_indices
chronological_split_indices = _policy.chronological_split_indices
completed_future_rows = _policy.completed_future_rows
completed_match_rows = _policy.completed_match_rows
is_market_feature = _policy.is_market_feature
is_prematch_feature = _policy.is_prematch_feature
no_odds_feature_columns = _policy.no_odds_feature_columns
parse_match_dates = _policy.parse_match_dates
prematch_feature_columns = _policy.prematch_feature_columns
prior_group_rolling = _policy.prior_group_rolling

__all__ = [
    "FEATURE_POLICY_VERSION",
    "chronological_partition_indices",
    "chronological_split_indices",
    "completed_future_rows",
    "completed_match_rows",
    "is_market_feature",
    "is_prematch_feature",
    "no_odds_feature_columns",
    "parse_match_dates",
    "prematch_feature_columns",
    "prior_group_rolling",
]
