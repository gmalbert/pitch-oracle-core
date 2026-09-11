"""Walk-forward evaluation harness for goal models.

Replaces the legacy "train on everything, evaluate on everything" pattern
with a proper sliding-window evaluation that respects temporal ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from penaltyblog.metrics import ignorance_score, multiclass_brier_score, rps_average


@dataclass(frozen=True)
class WalkForwardFold:
    """Results from one walk-forward fold."""
    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    n_train: int
    n_test: int
    n_scored: int
    n_skipped: int
    brier: float
    log_loss: float
    rps: float
    skip_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregated walk-forward evaluation."""
    folds: list[WalkForwardFold]
    window: int
    horizon: int

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def mean_brier(self) -> float:
        return float(np.mean([f.brier for f in self.folds]))

    @property
    def mean_log_loss(self) -> float:
        return float(np.mean([f.log_loss for f in self.folds]))

    @property
    def mean_rps(self) -> float:
        return float(np.mean([f.rps for f in self.folds]))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "fold": f.fold_index,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "n_train": f.n_train,
                "n_test": f.n_test,
                "n_scored": f.n_scored,
                "n_skipped": f.n_skipped,
                "skip_reasons": list(f.skip_reasons),
                "brier": f.brier,
                "log_loss": f.log_loss,
                "rps": f.rps,
            }
            for f in self.folds
        ])


def walk_forward_evaluate(
    matches: pd.DataFrame,
    model_factory: Callable[[pd.DataFrame], Any],
    *,
    window: int = 1500,
    horizon: int = 50,
    home_col: str = "team_home",
    away_col: str = "team_away",
    goals_home_col: str = "goals_home",
    goals_away_col: str = "goals_away",
    date_col: str = "datetime",
) -> WalkForwardResult:
    """Run a sliding-window walk-forward evaluation.

    ``model_factory`` receives a training DataFrame and must return a fitted
    model with a ``predict(home, away, max_goals)`` method returning a
    ``FootballProbabilityGrid``.
    """
    resolved_date_col = next(
        (candidate for candidate in (date_col, "datetime", "date", "kickoff_utc")
         if candidate in matches.columns),
        None,
    )
    if resolved_date_col is None:
        raise ValueError("matches require datetime, date, or kickoff_utc")
    df = matches.copy()
    df["_evaluation_date"] = pd.to_datetime(
        df[resolved_date_col], utc=True, errors="raise"
    )
    df = df.sort_values("_evaluation_date", kind="stable").reset_index(drop=True)
    folds: list[WalkForwardFold] = []
    fold_idx = 0

    if window <= 0 or horizon <= 0:
        raise ValueError("window and horizon must be positive")
    for start in range(0, len(df) - window - horizon + 1, horizon):
        train = df.iloc[start:start + window]
        test = df.iloc[start + window:start + window + horizon]
        model = model_factory(train)

        probs = []
        outcomes = []
        skip_reasons: list[str] = []
        for _, row in test.iterrows():
            try:
                grid = model.predict(row[home_col], row[away_col], max_goals=10)
                probs.append([grid.home_win, grid.draw, grid.away_win])
                gh, ga = int(row[goals_home_col]), int(row[goals_away_col])
                outcomes.append(0 if gh > ga else (1 if gh == ga else 2))
            except (KeyError, ValueError) as exc:
                skip_reasons.append(f"{type(exc).__name__}: {exc}")
                continue

        if not probs:
            raise ValueError(
                f"walk-forward fold {fold_idx} scored zero fixtures: {skip_reasons[:3]}"
            )

        p = np.array(probs)
        y = np.array(outcomes)
        folds.append(WalkForwardFold(
            fold_index=fold_idx,
            train_start=start,
            train_end=start + window,
            test_start=start + window,
            test_end=start + window + horizon,
            n_train=len(train),
            n_test=len(test),
            n_scored=len(probs),
            n_skipped=len(test) - len(probs),
            brier=float(multiclass_brier_score(p, y)),
            log_loss=float(ignorance_score(p, y)),
            rps=float(rps_average(p, y)),
            skip_reasons=tuple(sorted(set(skip_reasons))),
        ))
        fold_idx += 1

    return WalkForwardResult(folds=folds, window=window, horizon=horizon)
