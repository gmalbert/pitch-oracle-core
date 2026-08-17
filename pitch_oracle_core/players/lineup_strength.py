"""Expected-lineup strength bridge with explicit coverage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PlayerStrength:
    player_id: str
    attack_per_90: float
    defense_per_90: float
    effective_minutes: float
    estimated_at: datetime
    model_id: str

    def __post_init__(self) -> None:
        if self.effective_minutes < 0 or self.estimated_at.tzinfo is None:
            raise ValueError("invalid player strength state")


@dataclass(frozen=True)
class LineupMember:
    player_id: str
    expected_minutes: float
    availability_probability: float


def shrink_player(value: float, minutes: float, prior_minutes: float = 900.0) -> float:
    if minutes < 0 or prior_minutes <= 0:
        raise ValueError("invalid minutes")
    return value * minutes / (minutes + prior_minutes)


def lineup_delta(
    members: list[LineupMember],
    strengths: dict[str, PlayerStrength],
    replacement_attack_per_90: float,
    replacement_defense_per_90: float,
) -> tuple[float, float, float]:
    attack = defense = covered_minutes = 0.0
    for member in members:
        if not 0 <= member.availability_probability <= 1:
            raise ValueError("availability probability outside [0, 1]")
        expected = member.expected_minutes * member.availability_probability
        if not 0 <= expected <= 90:
            raise ValueError("expected player minutes outside [0, 90]")
        strength = strengths.get(member.player_id)
        if strength is None:
            continue
        attack += expected / 90 * shrink_player(
            strength.attack_per_90, strength.effective_minutes
        )
        defense += expected / 90 * shrink_player(
            strength.defense_per_90, strength.effective_minutes
        )
        covered_minutes += expected
    replacement_minutes = max(0.0, 990.0 - covered_minutes)
    attack += replacement_minutes / 90 * replacement_attack_per_90
    defense += replacement_minutes / 90 * replacement_defense_per_90
    return attack, defense, covered_minutes / 990.0
