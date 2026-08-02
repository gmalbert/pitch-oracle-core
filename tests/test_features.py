import pandas as pd
import pytest

from pitch_oracle_core.features import (
    chronological_split_indices,
    is_prematch_feature,
    prematch_feature_columns,
    prior_group_rolling,
)


def test_feature_policy_removes_postmatch_and_full_sample_leakage():
    frame = pd.DataFrame(columns=[
        "HomeShots", "FullTimeResult", "HalfTimeHomeWin", "HomeGoalsAve",
        "RefHomeWinRate", "Pinnacle_ClosingHomeOdds", "HomeMomentum_L3",
        "HomeRestDays", "ImpliedProb_HomeWin_Norm", "HomePointsZScore",
        "API_Home_Standing_Rank", "API_StandingsRankDiff",
    ])

    assert prematch_feature_columns(frame) == [
        "HomeMomentum_L3", "HomeRestDays", "ImpliedProb_HomeWin_Norm"
    ]
    assert not is_prematch_feature("AwayYellowCards")
    assert is_prematch_feature("AwayxG_Avg_L5")


def test_prior_group_rolling_excludes_current_row_and_other_clubs():
    frame = pd.DataFrame({
        "Team": ["A", "B", "A", "B", "A"],
        "Points": [3, 1, 0, 3, 1],
    })

    result = prior_group_rolling(
        frame, group="Team", value="Points", window=2, aggregation="sum"
    )

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 3
    assert result.iloc[3] == 1
    assert result.iloc[4] == 3


def test_temporal_split_never_trains_on_future_or_splits_a_match_date():
    dates = ["2025-01-03", "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
    train, test = chronological_split_indices(dates, test_size=0.4)
    parsed = pd.to_datetime(pd.Series(dates))

    assert parsed.iloc[train].max() < parsed.iloc[test].min()
    assert set(train).isdisjoint(test)


@pytest.mark.parametrize("test_size", [0, 1, -0.1])
def test_temporal_split_rejects_invalid_size(test_size):
    with pytest.raises(ValueError):
        chronological_split_indices(["2025-01-01", "2025-01-02"], test_size=test_size)
