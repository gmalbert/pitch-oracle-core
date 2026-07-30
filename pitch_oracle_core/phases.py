"""Phase transitions and opponent eligibility shared by league models."""

from collections.abc import Mapping, Sequence
from math import ceil, floor

from .config import LeagueConfig


def assign_phase(round_number: int, config: LeagueConfig) -> str:
    if config.phase.split_after_round is not None and round_number > config.phase.split_after_round:
        return "split"
    return "regular"


def apply_phase_transition(points: Mapping[str, float], config: LeagueConfig) -> dict[str, float]:
    """Return phase-start points, applying configured Belgian-style halving."""
    if not config.phase.points_halving:
        return dict(points)
    result = {}
    for team, value in points.items():
        if config.phase.points_halving_rounding == "ceil":
            result[team] = ceil(value / 2)
        elif config.phase.points_halving_rounding == "floor":
            result[team] = floor(value / 2)
        else:
            result[team] = value / 2
    return result


def eligible_opponents(team: str, phase: str, pools: Mapping[str, Sequence[str]] | None = None) -> set[str] | None:
    """Return allowed opponents, or ``None`` when the phase is unrestricted."""
    if phase == "regular" or pools is None:
        return None
    for members in pools.values():
        if team in members:
            return set(members) - {team}
    raise ValueError(f"Team {team!r} is not present in the configured phase pools")

