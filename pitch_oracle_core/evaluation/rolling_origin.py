"""Immutable rolling-origin evaluation rows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from pitch_oracle_core.evaluation.scores import outcome_index
from pitch_oracle_core.models.protocol import (
    FixtureFeatures,
    ScoreModel,
    validate_fixture_track,
)


LEDGER_KEY = ("fixture_id", "issued_at", "model_id")
RESULT_COLUMNS = {
    "actual_outcome", "actual_home_goals", "actual_away_goals",
    "result_ingested_at", "closing_price", "closing_line_value",
}


def append_forecast_ledger(
    existing: pd.DataFrame,
    additions: pd.DataFrame,
) -> pd.DataFrame:
    """Append issued forecasts and allow each result field to be filled only once."""
    missing = set(LEDGER_KEY).difference(additions.columns)
    if missing:
        raise ValueError(f"Forecast ledger additions miss: {sorted(missing)}")
    new = additions.copy()
    new["issued_at"] = pd.to_datetime(new["issued_at"], utc=True)
    if "kickoff_utc" in new:
        new["kickoff_utc"] = pd.to_datetime(new["kickoff_utc"], utc=True)
        if (new.issued_at >= new.kickoff_utc).any():
            raise ValueError("forecast must be issued before kickoff")
    if new.duplicated(list(LEDGER_KEY)).any():
        raise ValueError("ledger additions contain duplicate immutable keys")
    if existing.empty:
        return new.sort_values(list(LEDGER_KEY)).reset_index(drop=True)
    current = existing.copy()
    current["issued_at"] = pd.to_datetime(current["issued_at"], utc=True)
    if current.duplicated(list(LEDGER_KEY)).any():
        raise ValueError("existing ledger contains duplicate immutable keys")
    current_indexed = current.set_index(list(LEDGER_KEY), drop=False)
    append_rows = []
    for _, row in new.iterrows():
        key = tuple(row[column] for column in LEDGER_KEY)
        if key not in current_indexed.index:
            append_rows.append(row)
            continue
        old = current_indexed.loc[key]
        immutable_columns = (
            set(current.columns).intersection(new.columns).difference(RESULT_COLUMNS)
            .difference(LEDGER_KEY)
        )
        for column in immutable_columns:
            before, after = old[column], row[column]
            equal = (pd.isna(before) and pd.isna(after)) or before == after
            if not equal:
                raise ValueError(f"immutable forecast field changed: {column}")
        for column in RESULT_COLUMNS.intersection(new.columns):
            after = row[column]
            if pd.isna(after):
                continue
            before = old[column] if column in current else pd.NA
            if pd.notna(before) and before != after:
                raise ValueError(f"result field was already finalized: {column}")
            current_indexed.loc[key, column] = after
    result = current_indexed.reset_index(drop=True)
    if append_rows:
        result = pd.concat([result, pd.DataFrame(append_rows)], ignore_index=True)
    return result.sort_values(list(LEDGER_KEY)).reset_index(drop=True)


@dataclass(frozen=True)
class EvaluationFold:
    fold_id: str
    train_cutoff_utc: datetime
    test_end_utc: datetime

    def __post_init__(self) -> None:
        if self.train_cutoff_utc.tzinfo is None or self.test_end_utc.tzinfo is None:
            raise ValueError("fold timestamps must be timezone-aware")
        if self.test_end_utc <= self.train_cutoff_utc:
            raise ValueError("fold test end must follow train cutoff")


def evaluate_candidate(
    model_factory: Callable[[], ScoreModel],
    matches: pd.DataFrame,
    folds: Iterable[EvaluationFold],
    feature_builder: Callable[[pd.Series, datetime], FixtureFeatures],
) -> pd.DataFrame:
    frame = matches.copy()
    frame["kickoff_utc"] = pd.to_datetime(frame.kickoff_utc, utc=True)
    rows: list[dict[str, object]] = []
    for fold in folds:
        cutoff = pd.Timestamp(fold.train_cutoff_utc)
        end = pd.Timestamp(fold.test_end_utc)
        train = frame.loc[frame.kickoff_utc < cutoff].copy()
        test = frame.loc[
            (frame.kickoff_utc >= cutoff) & (frame.kickoff_utc < end)
        ].sort_values(["kickoff_utc", "fixture_id"])
        if test.empty:
            continue
        model = model_factory()
        model.spec.validate()
        model.fit(train, cutoff_utc=cutoff.to_pydatetime())
        for _, fixture in test.iterrows():
            features = feature_builder(fixture, cutoff.to_pydatetime())
            validate_fixture_track(model.spec, features)
            if pd.Timestamp(features.kickoff_utc) != fixture.kickoff_utc:
                raise ValueError("feature fixture kickoff does not match evaluation row")
            if pd.Timestamp(cutoff) >= fixture.kickoff_utc:
                raise ValueError("forecast issue time must precede kickoff")
            grid = model.predict_grid(features)
            probability = grid.normalized_one_x_two()
            actual = int(outcome_index(
                np.array([fixture.home_goals]), np.array([fixture.away_goals])
            )[0])
            rows.append({
                "fold_id": fold.fold_id,
                "fixture_id": fixture.fixture_id,
                "kickoff_utc": fixture.kickoff_utc,
                "issued_at": cutoff,
                "model_id": model.spec.model_id,
                "model_family": model.spec.family,
                "forecast_track": model.spec.track.value,
                "p_home": probability[0],
                "p_draw": probability[1],
                "p_away": probability[2],
                "tail_mass": grid.tail_mass,
                "actual_outcome": actual,
                "actual_home_goals": int(fixture.home_goals),
                "actual_away_goals": int(fixture.away_goals),
            })
    result = pd.DataFrame(rows)
    if not result.empty and result.duplicated(["model_id", "fixture_id"]).any():
        raise ValueError("a fixture was evaluated more than once for a model")
    return result
