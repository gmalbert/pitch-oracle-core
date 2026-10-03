"""Explicit PitchAPI feature families and league-specific promotion settings."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .advanced_team import METRICS, feature_names

FAMILIES = ("xg", "shot_profile", "advanced_team", "style", "player_strength", "predicted_lineup", "confirmed_lineup", "goalkeeper")
STYLE_COLUMNS = {"field_tilt_gap", "direct_speed_gap", "home_press_vs_away_buildup", "away_press_vs_home_buildup", "home_territory_pressure", "away_territory_pressure", "home_creation_quality", "away_creation_quality"}
TEAM_COLUMNS = ("team_player_attack", "team_player_defense", "team_player_coverage")
LINEUP_COLUMNS = ("lineup_attack_delta", "lineup_defense_delta", "lineup_coverage", "lineup_continuity", "lineup_missing_starter_value", "lineup_bench_attack", "lineup_bench_defense", "lineup_is_confirmed")
CONFIRMED_COLUMNS = ("confirmed_lineup_attack_delta", "confirmed_lineup_defense_delta", "confirmed_lineup_coverage")
KEEPER_COLUMNS = ("keeper_strength", "keeper_effective_minutes", "keeper_shots_on_target_faced", "keeper_start_probability", "keeper_claim_rate", "keeper_distribution_accuracy", "keeper_vaep_defensive", "keeper_sweeper_actions")

_FAMILY_COLUMNS = {family: set(feature_names(tuple(metric for metric in METRICS if metric.family == family))) for family in FAMILIES}
_FAMILY_COLUMNS["style"] = STYLE_COLUMNS
for family, columns in (("player_strength", TEAM_COLUMNS), ("predicted_lineup", LINEUP_COLUMNS), ("confirmed_lineup", CONFIRMED_COLUMNS), ("goalkeeper", KEEPER_COLUMNS)):
    _FAMILY_COLUMNS[family] = {f"{side}_{name}" for side in ("home", "away") for name in columns}
# Compatibility aliases use the same observation-aware xG state.
_FAMILY_COLUMNS["xg"].update(f"{side}_{metric}_ewm10" for side in ("home", "away") for metric in ("shot_quality", "finishing_vs_expectation"))
_COLUMN_FAMILY = {name: family for family, columns in _FAMILY_COLUMNS.items() for name in columns}


def feature_family(column: str) -> str | None:
    return _COLUMN_FAMILY.get(str(column))


def family_columns(family: str) -> tuple[str, ...]:
    if family not in _FAMILY_COLUMNS:
        raise ValueError(f"Unknown PitchAPI feature family: {family}")
    return tuple(sorted(_FAMILY_COLUMNS[family]))


@dataclass(frozen=True)
class FeatureFamilyConfig:
    """No family is promoted by merely collecting its data."""

    league_key: str
    enabled_families: tuple[str, ...] = ()
    evidence_id: str | None = None
    version: int = 1

    def __post_init__(self):
        if self.version != 1 or not self.league_key:
            raise ValueError("Invalid feature-family configuration")
        unknown = set(self.enabled_families).difference(FAMILIES)
        if unknown or len(set(self.enabled_families)) != len(self.enabled_families):
            raise ValueError(f"Invalid feature families: {sorted(unknown)}")
        if self.enabled_families and not self.evidence_id:
            raise ValueError("Promoted families require a validation evidence ID")

    @classmethod
    def load(cls, path: str | Path, *, league_key: str) -> "FeatureFamilyConfig":
        path = Path(path)
        if not path.exists():
            return cls(league_key)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("league_key") != league_key:
            raise ValueError("Feature promotion belongs to a different league")
        return cls(league_key, tuple(payload.get("enabled_families", ())), payload.get("evidence_id"), int(payload.get("version", -1)))

    def as_dict(self) -> dict:
        return {"version": self.version, "league_key": self.league_key, "enabled_families": list(self.enabled_families), "evidence_id": self.evidence_id}
