"""Basic unit tests for the Poisson evaluation functionality."""

import math

import pandas as pd
import pytest

from models.poisson_evaluation import evaluate_poisson_file, walk_forward_expectations


def test_walk_forward_forecasts_do_not_see_future_results():
    base = pd.DataFrame([
        {"MatchDate": "2025-01-01", "HomeTeam": "A", "AwayTeam": "B", "FullTimeHomeGoals": 1, "FullTimeAwayGoals": 0},
        {"MatchDate": "2025-01-02", "HomeTeam": "B", "AwayTeam": "A", "FullTimeHomeGoals": 1, "FullTimeAwayGoals": 1},
        {"MatchDate": "2025-01-03", "HomeTeam": "A", "AwayTeam": "B", "FullTimeHomeGoals": 2, "FullTimeAwayGoals": 0},
    ])
    changed_future = base.copy()
    changed_future.loc[2, ["FullTimeHomeGoals", "FullTimeAwayGoals"]] = [0, 8]

    original_home, original_away, _ = walk_forward_expectations(base)
    changed_home, changed_away, _ = walk_forward_expectations(changed_future)

    assert original_home.tolist() == pytest.approx(changed_home.tolist())
    assert original_away.tolist() == pytest.approx(changed_away.tolist())

def test_poisson_metrics_exist(tmp_path):
    matches = pd.DataFrame([
        {"MatchDate": "2025-01-01", "HomeTeam": "A", "AwayTeam": "B", "FullTimeHomeGoals": 1, "FullTimeAwayGoals": 0, "FullTimeResult": "H"},
        {"MatchDate": "2025-01-02", "HomeTeam": "B", "AwayTeam": "A", "FullTimeHomeGoals": 1, "FullTimeAwayGoals": 1, "FullTimeResult": "D"},
        {"MatchDate": "2025-01-03", "HomeTeam": "A", "AwayTeam": "B", "FullTimeHomeGoals": 2, "FullTimeAwayGoals": 0, "FullTimeResult": "H"},
        {"MatchDate": "2025-01-04", "HomeTeam": "B", "AwayTeam": "A", "FullTimeHomeGoals": 0, "FullTimeAwayGoals": 2, "FullTimeResult": "A"},
    ])
    fixture = tmp_path / "matches.csv"
    matches.to_csv(fixture, sep="\t", index=False)
    metrics = evaluate_poisson_file(str(fixture))
    # Expected keys
    expected = [
        'league_avg', 'home_mae', 'away_mae', 'home_rmse', 'away_rmse',
        'outcome_acc', 'brier_home', 'brier_draw', 'brier_away'
    ]
    for key in expected:
        assert key in metrics, f"Missing metric {key}"
        assert not math.isnan(metrics[key]), f"Metric {key} is NaN"
    # Basic sanity: MAE should be >=0 and less than league_avg*3
    assert metrics['home_mae'] >= 0
    assert metrics['away_mae'] >= 0
    assert metrics['home_rmse'] >= metrics['home_mae']
    assert metrics['away_rmse'] >= metrics['away_mae']
    assert 0 <= metrics['outcome_acc'] <= 1
    # Brier scores between 0 and 1
    assert 0 <= metrics['brier_home'] <= 1
    assert 0 <= metrics['brier_draw'] <= 1
    assert 0 <= metrics['brier_away'] <= 1
