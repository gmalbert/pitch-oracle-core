"""League-neutral split, points-transition, and bracket-path primitives."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, floor
from typing import Mapping, Sequence
from datetime import datetime
import pandas as pd

from pitch_oracle_core.domain.competitions import BracketRule, CompetitionRules, PhaseRule


@dataclass(frozen=True)
class PhaseTransition:
    phase_id: str
    pools: dict[str, tuple[str, ...]]
    starting_points: dict[str, float]
    brackets: tuple[BracketRule, ...]


def phase_at_issue(
    fixtures: pd.DataFrame,
    rules: CompetitionRules,
    issued_at: datetime,
) -> str:
    """Resolve the edition phase using only rounds completed before issuance."""
    if issued_at.tzinfo is None:
        raise ValueError("issued_at must be timezone-aware")
    required = {"kickoff_utc", "round", "status"}
    missing = required.difference(fixtures.columns)
    if missing:
        raise ValueError(f"phase fixtures miss: {sorted(missing)}")
    frame = fixtures.copy()
    frame["kickoff_utc"] = pd.to_datetime(frame.kickoff_utc, utc=True, errors="coerce")
    eligible = frame.loc[
        (frame.kickoff_utc < pd.Timestamp(issued_at))
        & frame.status.astype(str).str.lower().isin({"completed", "final"})
    ]
    numeric_rounds = pd.to_numeric(eligible["round"], errors="coerce").dropna()
    completed_round = int(numeric_rounds.max()) if not numeric_rounds.empty else 0
    selected = rules.phases[0].phase_id
    for phase in rules.phases:
        if phase.starts_after_round is not None and completed_round >= phase.starts_after_round:
            selected = phase.phase_id
    return selected


def transition_phase(
    ranking: Sequence[str],
    points: Mapping[str, float],
    rule: PhaseRule,
) -> PhaseTransition:
    if len(ranking) != len(set(ranking)) or set(ranking) != set(points):
        raise ValueError("ranking must contain every team exactly once")
    sizes = rule.pool_sizes or (len(ranking),)
    labels = rule.pool_labels or (rule.phase_id,)
    if len(sizes) != len(labels) or sum(sizes) != len(ranking):
        raise ValueError("phase pools must cover the ranking exactly")
    pools: dict[str, tuple[str, ...]] = {}
    offset = 0
    for label, size in zip(labels, sizes):
        pools[label] = tuple(ranking[offset:offset + size])
        offset += size

    def adjust(value: float) -> float:
        transitioned = float(value) * rule.points_multiplier
        if rule.points_rounding == "ceil":
            return float(ceil(transitioned))
        if rule.points_rounding == "floor":
            return float(floor(transitioned))
        return transitioned

    return PhaseTransition(
        phase_id=rule.phase_id,
        pools=pools,
        starting_points={team: adjust(points[team]) for team in ranking},
        brackets=rule.brackets,
    )


def generate_pool_fixtures(
    transition: PhaseTransition,
    *,
    repeats: int = 1,
) -> list[dict[str, object]]:
    if repeats < 1:
        raise ValueError("fixture repeats must be positive")
    rows = []
    for pool, teams in transition.pools.items():
        for first, second in combinations(teams, 2):
            for repeat in range(repeats):
                home, away = (first, second) if repeat % 2 == 0 else (second, first)
                rows.append({
                    "phase_id": transition.phase_id,
                    "pool": pool,
                    "home_team_id": home,
                    "away_team_id": away,
                    "repeat": repeat + 1,
                })
    return rows


def bracket_paths(transition: PhaseTransition) -> list[dict[str, object]]:
    return [
        {
            "phase_id": transition.phase_id,
            "bracket_id": bracket.bracket_id,
            "sources": list(bracket.sources),
            "legs": bracket.legs,
            "outcome_label": bracket.outcome_label,
        }
        for bracket in transition.brackets
    ]
