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


def build_split_pools(
    standings: Mapping[str, float], config: LeagueConfig
) -> dict[str, tuple[str, ...]]:
    """Partition regular-season standings into configured post-split pools.

    Teams are ranked by their supplied standings values (highest first), with
    the team name as a deterministic tie-breaker. Pool labels and sizes live
    in ``PhaseConfig`` so Scotland, Belgium, and future leagues share the same
    transition logic.
    """
    labels = config.phase.split_pools
    sizes = config.phase.split_pool_sizes
    if not labels:
        return {}
    if len(labels) != len(sizes) or sum(sizes) != len(standings):
        raise ValueError("split_pools and split_pool_sizes must cover every team exactly once")
    if any(size < 1 for size in sizes):
        raise ValueError("split pool sizes must be positive")
    ranked = sorted(standings, key=lambda team: (-standings[team], str(team)))
    pools: dict[str, tuple[str, ...]] = {}
    offset = 0
    for label, size in zip(labels, sizes):
        pools[label] = tuple(ranked[offset:offset + size])
        offset += size
    return pools


def phase_start_standings(
    regular_standings: Mapping[str, float], config: LeagueConfig
) -> dict[str, float]:
    """Apply configured points transition and sanctions at a phase boundary."""
    transitioned = apply_phase_transition(regular_standings, config)
    for team, adjustment in config.points_adjustments.items():
        if team in transitioned:
            transitioned[team] += adjustment
    return transitioned


def eligible_opponents(team: str, phase: str, pools: Mapping[str, Sequence[str]] | None = None) -> set[str] | None:
    """Return allowed opponents, or ``None`` when the phase is unrestricted."""
    if phase == "regular" or pools is None:
        return None
    for members in pools.values():
        if team in members:
            return set(members) - {team}
    raise ValueError(f"Team {team!r} is not present in the configured phase pools")

