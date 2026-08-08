"""Reproducible diagnostics for deciding whether a forecast model is deployable."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import (
    completed_future_rows,
    is_market_feature,
    no_odds_feature_columns,
    prematch_feature_columns,
)


CLASS_LABELS = (0, 1, 2)
TARGET_MAP = {"H": 0, "D": 1, "A": 2}


@dataclass
class TemperatureScaledClassifier:
    """Serializable inference wrapper for validation-period temperature scaling."""

    estimator: Any
    temperature: float

    @property
    def n_features_in_(self) -> int | None:
        return getattr(self.estimator, "n_features_in_", None)

    @property
    def classes_(self) -> Any:
        return getattr(self.estimator, "classes_", np.asarray(CLASS_LABELS))

    def predict_proba(self, X: Any) -> np.ndarray:
        probabilities = normalize_probabilities(self.estimator.predict_proba(X))
        logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        scaled = np.exp(logits)
        return scaled / scaled.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def fit_temperature(probabilities: np.ndarray, y_true: Sequence[int]) -> float:
    """Fit one calibration parameter on a held-out chronological period."""
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import log_loss

    probabilities = normalize_probabilities(probabilities)
    labels = np.asarray(y_true, dtype=int)

    def objective(temperature: float) -> float:
        logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        scaled = np.exp(logits)
        scaled /= scaled.sum(axis=1, keepdims=True)
        return float(log_loss(labels, scaled, labels=CLASS_LABELS))

    result = minimize_scalar(objective, bounds=(0.25, 4.0), method="bounded")
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("Temperature calibration failed")
    return float(result.x)


def normalize_probabilities(values: Sequence[Sequence[float]]) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("Expected a probability matrix with three outcome columns")
    totals = probabilities.sum(axis=1, keepdims=True)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (totals <= 0).any()
    ):
        raise ValueError("Probabilities must be finite, non-negative and have positive mass")
    return probabilities / totals


def multiclass_brier_score(y_true: Sequence[int], probabilities: np.ndarray) -> float:
    probabilities = normalize_probabilities(probabilities)
    targets = np.eye(3, dtype=float)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def expected_calibration_error(
    y_true: Sequence[int], probabilities: np.ndarray, *, bins: int = 10
) -> float:
    """Top-label ECE: the confidence/accuracy gap weighted by bin population."""
    if bins < 2:
        raise ValueError("bins must be at least two")
    probabilities = normalize_probabilities(probabilities)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == np.asarray(y_true, dtype=int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(correct)
    if total == 0:
        raise ValueError("At least one observation is required")
    error = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if upper_inclusive else confidence < edges[index + 1]
        )
        if mask.any():
            error += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return error


def probability_metrics(y_true: Sequence[int], probabilities: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, log_loss, recall_score

    probabilities = normalize_probabilities(probabilities)
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "log_loss": float(log_loss(y_true, probabilities, labels=CLASS_LABELS)),
        "brier_score": multiclass_brier_score(y_true, probabilities),
        "calibration_error": expected_calibration_error(y_true, probabilities),
        "draw_recall": float(
            recall_score(y_true, predicted, labels=[1], average=None, zero_division=0)[0]
        ),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "decisive_lean_rate": float((probabilities.max(axis=1) >= 0.55).mean()),
    }


def rolling_origin_splits(
    dates: Sequence[object], *, n_splits: int = 3, minimum_train_fraction: float = 0.5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding-window folds with contiguous, non-overlapping tests."""
    parsed = pd.to_datetime(pd.Series(dates), errors="coerce")
    if parsed.isna().any():
        raise ValueError("Rolling-origin validation requires a valid date on every row")
    unique_dates = np.array(sorted(parsed.unique()))
    minimum_dates = max(1, int(np.ceil(len(unique_dates) * minimum_train_fraction)))
    available = len(unique_dates) - minimum_dates
    if n_splits < 1 or available < n_splits:
        raise ValueError("Not enough distinct dates for the requested rolling-origin folds")
    blocks = np.array_split(unique_dates[minimum_dates:], n_splits)
    folds = []
    for block in blocks:
        cutoff = block[0]
        end = block[-1]
        train = np.flatnonzero((parsed < cutoff).to_numpy())
        test = np.flatnonzero(((parsed >= cutoff) & (parsed <= end)).to_numpy())
        if len(train) and len(test):
            folds.append((train, test))
    return folds


def feature_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    prematch = set(prematch_feature_columns(frame))
    rows = []
    for name in frame.columns:
        if name not in prematch:
            category = "excluded"
        elif is_market_feature(name):
            category = "market"
        else:
            category = "no_odds"
        rows.append({
            "feature": name,
            "category": category,
            "dtype": str(frame[name].dtype),
            "missing_rate": float(frame[name].isna().mean()),
            "unique_values": int(frame[name].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class AblationResult:
    candidate: str
    feature_count: int
    folds: int
    test_rows: int
    metrics: dict[str, float]


def evaluate_feature_ablation(
    frame: pd.DataFrame,
    estimator_factory: Callable[[], Any],
    *,
    n_splits: int = 3,
) -> list[AblationResult]:
    valid = frame["FullTimeResult"].isin(TARGET_MAP)
    dates = pd.to_datetime(frame["MatchDate"], errors="coerce")
    valid &= dates.notna()
    working = frame.loc[valid].reset_index(drop=True)
    dates = dates.loc[valid].reset_index(drop=True)
    y = working["FullTimeResult"].map(TARGET_MAP).to_numpy()
    folds = rolling_origin_splits(dates, n_splits=n_splits)
    candidates = {
        "odds_heavy": prematch_feature_columns(working),
        "no_odds": no_odds_feature_columns(working),
    }
    results = []
    baseline_y, baseline_probabilities = [], []
    for train, test in folds:
        prior = np.bincount(y[train], minlength=3).astype(float)
        prior /= prior.sum()
        baseline_y.extend(y[test].tolist())
        baseline_probabilities.extend(np.tile(prior, (len(test), 1)).tolist())
    results.append(AblationResult(
        candidate="class_prior_baseline",
        feature_count=0,
        folds=len(folds),
        test_rows=len(baseline_y),
        metrics=probability_metrics(baseline_y, np.asarray(baseline_probabilities)),
    ))
    for candidate, columns in candidates.items():
        numeric_columns = list(working[columns].select_dtypes(include=[np.number]).columns)
        if not numeric_columns:
            raise ValueError(f"Candidate {candidate!r} has no numeric features")
        all_y, all_probabilities = [], []
        for train, test in folds:
            train_frame = working.iloc[train][numeric_columns]
            test_frame = working.iloc[test][numeric_columns]
            imputation = train_frame.mean().fillna(0.0)
            X_train = train_frame.fillna(imputation).fillna(0.0).to_numpy()
            X_test = test_frame.fillna(imputation).fillna(0.0).to_numpy()
            estimator = estimator_factory()
            estimator.fit(X_train, y[train])
            probabilities = normalize_probabilities(estimator.predict_proba(X_test))
            all_y.extend(y[test].tolist())
            all_probabilities.extend(probabilities.tolist())
        results.append(AblationResult(
            candidate=candidate,
            feature_count=len(numeric_columns),
            folds=len(folds),
            test_rows=len(all_y),
            metrics=probability_metrics(all_y, np.asarray(all_probabilities)),
        ))
    return results


def data_quality_summary(frame: pd.DataFrame, *, as_of: object | None = None) -> dict[str, Any]:
    dates = pd.to_datetime(frame.get("MatchDate"), errors="coerce")
    invalid_future = completed_future_rows(frame, as_of=as_of)
    no_odds = no_odds_feature_columns(frame)
    numeric_no_odds = frame[no_odds].select_dtypes(include=[np.number])
    return {
        "rows": int(len(frame)),
        "invalid_dates": int(dates.isna().sum()),
        "completed_future_rows": int(len(invalid_future)),
        "earliest_date": dates.min().isoformat() if dates.notna().any() else None,
        "latest_date": dates.max().isoformat() if dates.notna().any() else None,
        "duplicate_fixture_rows": int(
            frame.duplicated(subset=["MatchDate", "HomeTeam", "AwayTeam"], keep=False).sum()
        ),
        "duplicate_no_odds_feature_rows": int(numeric_no_odds.duplicated(keep=False).sum()),
    }
