import pytest
import pandas as pd

from pitch_oracle_core import calculate_goal_markets
from models.poisson_predictor import predict_match_poisson


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


@pytest.mark.parametrize("line", [0, 2.25, 4.0])
def test_only_standard_half_goal_lines_are_supported(line):
    with pytest.raises(ValueError):
        calculate_goal_markets(1.0, 1.0, lines=(line,))
