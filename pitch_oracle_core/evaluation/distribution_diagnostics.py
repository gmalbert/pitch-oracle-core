"""Auditable count-distribution diagnostics used to freeze candidate sets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import pandas as pd


@dataclass(frozen=True)
class GoalDiagnostics:
    fixtures: int
    home_mean: float
    home_variance: float
    away_mean: float
    away_variance: float
    home_dispersion_ratio: float
    away_dispersion_ratio: float
    zero_zero_rate: float
    draw_rate: float
    low_score_rate: float
    four_plus_goal_rate: float


def describe_goal_counts(matches: pd.DataFrame) -> GoalDiagnostics:
    required = {"home_goals", "away_goals"}
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    frame = matches.dropna(subset=list(required)).copy()
    if len(frame) < 2:
        raise ValueError("at least two completed fixtures are required")
    home = frame.home_goals.astype(int).to_numpy()
    away = frame.away_goals.astype(int).to_numpy()
    if (home < 0).any() or (away < 0).any():
        raise ValueError("goals cannot be negative")
    home_mean, away_mean = float(home.mean()), float(away.mean())
    home_variance, away_variance = float(home.var(ddof=1)), float(away.var(ddof=1))
    return GoalDiagnostics(
        fixtures=len(frame),
        home_mean=home_mean,
        home_variance=home_variance,
        away_mean=away_mean,
        away_variance=away_variance,
        home_dispersion_ratio=home_variance / max(home_mean, 1e-12),
        away_dispersion_ratio=away_variance / max(away_mean, 1e-12),
        zero_zero_rate=float(((home == 0) & (away == 0)).mean()),
        draw_rate=float((home == away).mean()),
        low_score_rate=float(((home <= 1) & (away <= 1)).mean()),
        four_plus_goal_rate=float(((home >= 4) | (away >= 4)).mean()),
    )


def diagnostics_by_edition(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (competition_id, edition_id), group in matches.groupby(
        ["competition_id", "edition_id"], sort=True
    ):
        rows.append({
            "competition_id": competition_id,
            "edition_id": edition_id,
            **asdict(describe_goal_counts(group)),
        })
    return pd.DataFrame(rows)
