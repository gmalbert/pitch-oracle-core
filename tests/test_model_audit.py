import numpy as np
import pandas as pd
import pytest

from pitch_oracle_core.model_audit import (
    data_quality_summary,
    expected_calibration_error,
    fit_temperature,
    feature_inventory,
    multiclass_brier_score,
    probability_metrics,
    rolling_origin_splits,
    TemperatureScaledClassifier,
)
from sklearn.dummy import DummyClassifier


def test_probability_metrics_reward_correct_calibrated_forecasts():
    y = [0, 1, 2]
    perfect = np.eye(3)

    assert multiclass_brier_score(y, perfect) == pytest.approx(0.0)
    assert expected_calibration_error(y, perfect) == pytest.approx(0.0)
    assert probability_metrics(y, perfect)["log_loss"] < 1e-10


def test_temperature_scaling_softens_overconfident_probabilities():
    y = [0, 1, 2, 0, 1, 2]
    overconfident = np.asarray([
        [0.99, 0.005, 0.005], [0.99, 0.005, 0.005], [0.005, 0.005, 0.99],
        [0.99, 0.005, 0.005], [0.005, 0.99, 0.005], [0.005, 0.005, 0.99],
    ])

    temperature = fit_temperature(overconfident, y)

    assert temperature > 1.0


def test_rolling_origin_folds_only_train_on_prior_dates():
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    folds = rolling_origin_splits(dates, n_splits=3)
    series = pd.Series(dates)

    assert len(folds) == 3
    for train, test in folds:
        assert series.iloc[train].max() < series.iloc[test].min()


def test_inventory_and_quality_expose_market_features_and_bad_dates():
    frame = pd.DataFrame([
        {"MatchDate": "2026-05-10", "HomeTeam": "A", "AwayTeam": "B", "FullTimeResult": "H", "PSCH": 1.8, "HomeMomentum_L3": 1.0},
        {"MatchDate": "2026-10-05", "HomeTeam": "B", "AwayTeam": "A", "FullTimeResult": "A", "PSCH": 2.2, "HomeMomentum_L3": 2.0},
    ])

    inventory = feature_inventory(frame).set_index("feature")
    quality = data_quality_summary(frame, as_of="2026-08-04")

    assert inventory.loc["PSCH", "category"] == "market"
    assert inventory.loc["HomeMomentum_L3", "category"] == "no_odds"
    assert quality["completed_future_rows"] == 1


def test_ablation_includes_a_fold_specific_class_prior_baseline():
    from pitch_oracle_core.model_audit import evaluate_feature_ablation

    rows = []
    labels = ["H", "D", "A"] * 20
    for index, label in enumerate(labels):
        rows.append({
            "MatchDate": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
            "FullTimeResult": label,
            "HomeMomentum_L3": float(index % 4),
        })
    results = evaluate_feature_ablation(
        pd.DataFrame(rows), lambda: DummyClassifier(strategy="prior"), n_splits=2
    )

    assert results[0].candidate == "class_prior_baseline"
    assert results[0].feature_count == 0


def test_ablation_includes_poisson_candidate_when_goals_are_present():
    from pitch_oracle_core.model_audit import evaluate_feature_ablation

    rows = []
    teams = ["A", "B"]
    labels = ["H", "D", "A"] * 20
    for index, label in enumerate(labels):
        home = teams[index % 2]
        away = teams[(index + 1) % 2]
        goals = (1, 0) if label == "H" else (0, 1) if label == "A" else (1, 1)
        rows.append({
            "MatchDate": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
            "FullTimeResult": label,
            "HomeTeam": home,
            "AwayTeam": away,
            "FullTimeHomeGoals": goals[0],
            "FullTimeAwayGoals": goals[1],
            "HomeMomentum_L3": float(index % 4),
        })
    results = evaluate_feature_ablation(
        pd.DataFrame(rows), lambda: DummyClassifier(strategy="prior"), n_splits=2
    )

    by_name = {result.candidate: result for result in results}
    assert "poisson" in by_name
    assert by_name["poisson"].test_rows > 0
    assert by_name["poisson"].metrics["log_loss"] > 0
    assert by_name["poisson"].metrics["brier_score"] >= 0


def test_ablation_poisson_candidate_skips_without_goal_columns():
    from pitch_oracle_core.model_audit import evaluate_feature_ablation

    rows = []
    labels = ["H", "D", "A"] * 20
    for index, label in enumerate(labels):
        rows.append({
            "MatchDate": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
            "FullTimeResult": label,
            "HomeMomentum_L3": float(index % 4),
        })
    results = evaluate_feature_ablation(
        pd.DataFrame(rows), lambda: DummyClassifier(strategy="prior"), n_splits=2
    )

    by_name = {result.candidate: result for result in results}
    assert "poisson" in by_name
    assert by_name["poisson"].test_rows == 0
