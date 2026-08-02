import pytest
import pandas as pd

from pitch_oracle_core import calculate_goal_markets
from models.poisson_predictor import PoissonPredictor, predict_match_poisson


def test_goal_markets_are_internally_consistent():
    markets = calculate_goal_markets(1.4, 1.1)

    assert markets.total_expected_goals == pytest.approx(2.5)
    assert markets.over_under[2.5][0] + markets.over_under[2.5][1] == pytest.approx(1.0)
    assert markets.btts_yes + markets.btts_no == pytest.approx(1.0)
    assert markets.most_likely_score == (1, 1)


def test_goal_markets_expose_legacy_keys():
    result = calculate_goal_markets(1.4, 1.1).as_dict()

    assert result["ExpectedTotalGoals"] == pytest.approx(2.5)
    assert result["Over2_5Prob"] + result["Under2_5Prob"] == pytest.approx(1.0)
    assert result["BTTSProb"] == result["BTTSYesProb"]


def test_poisson_predictor_includes_shared_goal_markets():
    stats = pd.DataFrame({
        "Team": ["Home FC", "Away FC"],
        "HomeGoalsAve": [1.4, 1.2],
        "AwayGoalsAve": [1.1, 1.3],
        "HomeGoalsConcededAve": [1.0, 1.2],
        "AwayGoalsConcededAve": [1.1, 1.0],
    })

    result = predict_match_poisson("Home FC", "Away FC", stats)

    assert result["ExpectedTotalGoals"] == pytest.approx(
        result["ExpectedHomeGoals"] + result["ExpectedAwayGoals"]
    )
    assert 0 <= result["Over2_5Prob"] <= 1
    assert 0 <= result["BTTSProb"] <= 1


def test_poisson_expected_goals_increase_against_a_leakier_defense():
    predictor = PoissonPredictor()
    tight_home, _ = predictor.estimate_goals(1.4, 1.0, 1.2, 0.8)
    leaky_home, _ = predictor.estimate_goals(1.4, 1.0, 1.2, 1.8)

    assert leaky_home > tight_home


def test_poisson_uses_the_correct_home_and_away_defensive_splits():
    stats = pd.DataFrame({
        "Team": ["Home FC", "Away FC"],
        "HomeGoalsAve": [1.4, 1.0],
        "AwayGoalsAve": [1.0, 1.4],
        "HomeGoalsConcededAve": [0.8, 0.6],
        "AwayGoalsConcededAve": [2.0, 1.8],
    })

    result = predict_match_poisson("Home FC", "Away FC", stats)
    assert result["ExpectedHomeGoals"] == pytest.approx(1.8)
    assert result["ExpectedAwayGoals"] == pytest.approx(0.8)


def test_truncated_score_grid_is_normalized_for_outcome_probabilities():
    predictor = PoissonPredictor()
    scorelines = predictor.poisson_scoreline_probabilities(3.5, 2.8, max_goals=2)
    probabilities = predictor.predict_match_outcome(scorelines)
    assert sum(probabilities) == pytest.approx(1.0)


@pytest.mark.parametrize("line", [0, 2.25, 4.0])
def test_only_standard_half_goal_lines_are_supported(line):
    with pytest.raises(ValueError):
        calculate_goal_markets(1.0, 1.0, lines=(line,))


@pytest.mark.parametrize("home,away", [(float("nan"), 1.0), (1.0, float("inf"))])
def test_goal_markets_reject_non_finite_rates(home, away):
    with pytest.raises(ValueError):
        calculate_goal_markets(home, away)
