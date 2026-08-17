"""Tests for the PitchAPI integration: config gate, fetcher, and xG feature merge."""

from pathlib import Path

import pandas as pd
import pytest

from pitch_oracle_core import get_league_config, PitchAPIXGProvider
from fetch_pitchapi import (
    PitchAPIError,
    _match_xg_row,
    _shot_rows,
    pitchapi_league_id,
)


def _match(home="Ajax", away="Telstar", home_id="t_1", away_id="t_2"):
    return {
        "id": "m_test",
        "date": "2025-08-10",
        "status": "finished",
        "home_team": {"id": home_id, "name": home},
        "away_team": {"id": away_id, "name": away},
        "score_home": 2,
        "score_away": 0,
    }


def _periods():
    return [
        {"period": "FirstHalf", "shots": [
            {"team_id": "t_1", "player": {"id": "p_1", "name": "P1"},
             "minute": 10, "x": 90.0, "y": 30.0, "expected_goals": 0.5,
             "is_on_target": True, "shot_type": "RightFoot", "situation": "RegularPlay",
             "event_type": "Goal", "is_inside_box": True},
            {"team_id": "t_2", "player": {"id": "p_2", "name": "P2"},
             "minute": 20, "x": 85.0, "y": 20.0, "expected_goals": 0.25,
             "is_on_target": False, "shot_type": "LeftFoot", "situation": "FromCorner",
             "event_type": "Miss", "is_inside_box": False},
        ]},
    ]


def test_all_target_leagues_enable_pitchapi_with_league_ids():
    for key, league_id in (
        ("epl", "l_4WFCIZ"), ("scotland", "l_1LMdEO"), ("eredivisie", "l_4H43wr"),
        ("portugal", "l_4QexZg"), ("belgium", "l_2L6d1F"), ("turkey", "l_0S1uaf"),
    ):
        config = get_league_config(key)
        assert config.sources.pitchapi
        assert config.sources.pitchapi_league_id == league_id


def test_pitchapi_league_id_requires_configuration():
    from pitch_oracle_core.config import DataSourceConfig, LeagueConfig
    unconfigured = LeagueConfig(
        "x", "X", "X0", None, None, 18, (8, 5),
        sources=DataSourceConfig(pitchapi=False),
    )
    with pytest.raises(ValueError):
        pitchapi_league_id(unconfigured)


def test_shot_rows_flatten_periods():
    rows = _shot_rows(_match(), _periods())
    assert len(rows) == 2
    assert rows[0]["match_id"] == "m_test"
    assert rows[0]["expected_goals"] == 0.5
    assert rows[0]["player_name"] == "P1"
    assert rows[1]["event_type"] == "Miss"


def test_match_xg_row_sums_per_side_and_normalizes_names():
    aliases = {"Ajax Amsterdam": "Ajax", "FC Groningen": "Groningen"}
    row = _match_xg_row(
        _match(home="Ajax Amsterdam", away="FC Groningen"), _periods(), aliases
    )
    assert row["home_xg"] == 0.5
    assert row["away_xg"] == 0.25
    # aliases normalize PitchAPI names to the football-data vocabulary
    assert row["HomeTeam"] == "Ajax"
    assert row["AwayTeam"] == "Groningen"


def test_match_xg_side_uses_nested_team_ids():
    row = _match_xg_row(_match(), _periods())
    assert row["home_team_id"] == "t_1"
    assert row["away_team_id"] == "t_2"


def test_prepare_merge_activates_ledger_xg_features(tmp_path: Path):
    from prepare_model_data import prepare_historical_features

    hist = pd.DataFrame([
        {"Date": "2025-08-09", "Time": "16:30", "HomeTeam": "Ajax", "AwayTeam": "Telstar",
         "FTHG": 2, "FTAG": 0, "FTR": "H", "HS": 15, "AS": 6, "HST": 8, "AST": 2},
        {"Date": "2025-08-16", "Time": "16:30", "HomeTeam": "Telstar", "AwayTeam": "Ajax",
         "FTHG": 1, "FTAG": 1, "FTR": "D", "HS": 10, "AS": 11, "HST": 4, "AST": 5},
    ])
    hist.to_csv(tmp_path / "combined_historical_data.csv", sep="\t", index=False)
    xg = pd.DataFrame([
        {"match_date": "2025-08-09", "HomeTeam": "Ajax", "AwayTeam": "Telstar",
         "home_xg": 2.4, "away_xg": 0.8},
        {"match_date": "2025-08-16", "HomeTeam": "Telstar", "AwayTeam": "Ajax",
         "home_xg": 1.1, "away_xg": 1.3},
    ])
    xg.to_csv(tmp_path / "pitchapi_match_xg.csv", sep="\t", index=False)

    result = prepare_historical_features(
        league_key="eredivisie",
        source=tmp_path / "combined_historical_data.csv",
        destination=tmp_path / "out.csv",
        xg_source=tmp_path / "pitchapi_match_xg.csv",
    )
    assert {"home_xg", "away_xg"} <= set(result.columns)
    assert {"home_xg_for_ewm10", "away_xg_for_ewm10"} <= set(result.columns)
    # xG values land on the right rows
    row0 = result.sort_values("MatchDate").iloc[0]
    assert row0["home_xg"] == pytest.approx(2.4)
    assert row0["away_xg"] == pytest.approx(0.8)


def test_prepare_without_xg_source_keeps_columns_absent(tmp_path: Path):
    from prepare_model_data import prepare_historical_features

    hist = pd.DataFrame([
        {"Date": "2025-08-09", "Time": "16:30", "HomeTeam": "Ajax", "AwayTeam": "Telstar",
         "FTHG": 2, "FTAG": 0, "FTR": "H", "HS": 15, "AS": 6, "HST": 8, "AST": 2},
    ])
    hist.to_csv(tmp_path / "combined_historical_data.csv", sep="\t", index=False)
    result = prepare_historical_features(
        league_key="eredivisie",
        source=tmp_path / "combined_historical_data.csv",
        destination=tmp_path / "out.csv",
        xg_source=tmp_path / "pitchapi_match_xg.csv",  # file does not exist
    )
    assert "home_xg" not in result.columns
    assert "home_xg_for_ewm10" not in result.columns


def test_pitchapi_error_carries_machine_code():
    exc = PitchAPIError("ANALYTICS_UNAVAILABLE: match exists but was never rated")
    assert "ANALYTICS_UNAVAILABLE" in str(exc)


def test_provider_export():
    assert PitchAPIXGProvider.__name__ == "PitchAPIXGProvider"
