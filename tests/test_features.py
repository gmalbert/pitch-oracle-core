import pandas as pd
import pytest

from pitch_oracle_core.features import (
    chronological_split_indices,
    chronological_partition_indices,
    completed_match_rows,
    completed_future_rows,
    is_market_feature,
    is_prematch_feature,
    no_odds_feature_columns,
    parse_match_dates,
    prematch_feature_columns,
    prior_group_rolling,
)


def test_completed_match_rows_rejects_undated_and_unfinished_fixtures():
    frame = pd.DataFrame(
        [
            {"MatchDate": "2026-08-01", "FullTimeResult": "H"},
            {"MatchDate": None, "FullTimeResult": "A"},
            {"MatchDate": "2026-08-03", "FullTimeResult": None},
        ]
    )

    result = completed_match_rows(frame)

    assert len(result) == 1
    assert result.loc[0, "FullTimeResult"] == "H"
    assert result.loc[0, "MatchDate"] == pd.Timestamp("2026-08-01")


def test_match_date_parser_preserves_iso_and_parses_source_dates_day_first():
    parsed = parse_match_dates(["2026-10-05", "10/05/2026", "13/05/2026"])

    assert parsed.tolist() == [
        pd.Timestamp("2026-10-05"),
        pd.Timestamp("2026-05-10"),
        pd.Timestamp("2026-05-13"),
    ]


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


def test_no_odds_policy_removes_descriptive_engineered_and_legacy_market_fields():
    frame = pd.DataFrame(columns=[
        "Bet365_HomeWinOdds", "ImpliedProb_HomeWin_Norm", "Bet365_Value_Home",
        "PSCH", "AvgCA", "BFEAHH", "1XBCH", "HomeMomentum_L3", "EloDiff",
    ])

    assert no_odds_feature_columns(frame) == ["HomeMomentum_L3", "EloDiff"]
    assert is_market_feature("Pinnacle_ClosingHomeOdds")
    assert is_market_feature("B365C>2.5")
    assert not is_market_feature("AwayPointsPerGame")


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


def test_temporal_partition_keeps_three_strictly_ordered_periods():
    dates = pd.date_range("2025-01-01", periods=20, freq="D").repeat(2)
    train, calibration, test = chronological_partition_indices(
        dates, calibration_size=0.2, test_size=0.2
    )
    parsed = pd.Series(dates)

    assert parsed.iloc[train].max() < parsed.iloc[calibration].min()
    assert parsed.iloc[calibration].max() < parsed.iloc[test].min()


def test_completed_future_rows_flags_corrupted_chronology():
    frame = pd.DataFrame([
        {"MatchDate": "2026-05-10", "FullTimeResult": "H"},
        {"MatchDate": "2026-10-05", "FullTimeResult": "A"},
        {"MatchDate": "2026-10-06", "FullTimeResult": None},
    ])

    invalid = completed_future_rows(frame, as_of="2026-08-04")

    assert invalid["MatchDate"].tolist() == ["2026-10-05"]


@pytest.mark.parametrize("test_size", [0, 1, -0.1])
def test_temporal_split_rejects_invalid_size(test_size):
    with pytest.raises(ValueError):
        chronological_split_indices(["2025-01-01", "2025-01-02"], test_size=test_size)
