import pandas as pd

from pitch_oracle_core.ui_pages import (
    _filter_predictions_by_risk,
    _display_predictions,
    _prediction_commentary,
    _style_prediction_cells,
    _style_prediction_risk,
    _team_deep_dive_dictionary,
    _team_match_log,
)


def test_prediction_risk_filter_selects_the_requested_band():
    predictions = pd.DataFrame({"Risk_Score": [20.0, 70.0, 85.0, 95.0]})

    assert _filter_predictions_by_risk(predictions, "Low risk").index.tolist() == [0]
    assert _filter_predictions_by_risk(predictions, "Moderate risk").index.tolist() == [1]
    assert _filter_predictions_by_risk(predictions, "High risk").index.tolist() == [2]
    assert _filter_predictions_by_risk(predictions, "Critical risk").index.tolist() == [3]


def test_prediction_row_styling_uses_risk_colors():
    columns = ["Home team", "Risk score"]

    assert "#d4edda" in _style_prediction_risk(pd.Series(["A", 20], index=columns))[0]
    assert "#fff3cd" in _style_prediction_risk(pd.Series(["A", 70], index=columns))[0]
    assert "#ffe5b4" in _style_prediction_risk(pd.Series(["A", 85], index=columns))[0]
    assert "#f8d7da" in _style_prediction_risk(pd.Series(["A", 95], index=columns))[0]


def test_prediction_cell_styling_bolds_actionable_team():
    row = pd.Series({
        "Home team": "Home FC", "Away team": "Away FC", "Risk score": 90,
        "Home win": "48.0%", "Draw": "30.0%", "Away win": "22.0%",
        "Recommendation": "No Clear Edge",
    })
    styles = _style_prediction_cells(row)
    assert "font-weight: bold !important" in styles[0]
    assert "font-weight" not in styles[1]


def test_prediction_commentary_includes_goal_markets():
    row = pd.Series({
        "HomeWin_Prob": 0.7, "Draw_Prob": 0.2, "AwayWin_Prob": 0.1,
        "HomeTeam": "Home FC", "AwayTeam": "Away FC", "Risk_Category": "Low Risk",
        "ExpectedTotalGoals": 2.345, "Over2_5Prob": 0.61,
        "Under2_5Prob": 0.39, "BTTSProb": 0.52,
    })
    commentary = _prediction_commentary(row)
    assert "Expected total goals: 2.35" in commentary
    assert "Over 2.5: 61.0%" in commentary
    assert "Under 2.5: 39.0%" in commentary
    assert "BTTS: 52.0%" in commentary
    assert "<br>" not in commentary


def test_prediction_commentary_includes_available_weather():
    row = pd.Series({
        "HomeWin_Prob": 0.45, "Draw_Prob": 0.30, "AwayWin_Prob": 0.25,
        "HomeTeam": "Ajax", "AwayTeam": "Twente", "Risk_Category": "Moderate Risk",
        "WeatherDescription": "Partly cloudy", "Temperature": 19.4,
        "Precipitation": 0.2, "WindSpeed": 3.1, "Humidity": 64,
    })

    commentary = _prediction_commentary(row)

    assert "Expected weather:** Partly cloudy · 19.4°C" in commentary
    assert "0.2 mm rain" in commentary


def test_prediction_commentary_explains_when_forecast_is_too_early():
    row = pd.Series({
        "HomeWin_Prob": 0.45, "Draw_Prob": 0.30, "AwayWin_Prob": 0.25,
        "HomeTeam": "Ajax", "AwayTeam": "Twente", "Risk_Category": "Moderate Risk",
        "Date": "2099-08-08",
    })

    assert "Expected weather:** Available closer to kickoff" in _prediction_commentary(row)


def test_display_predictions_marks_highest_probability_team():
    display = _display_predictions(pd.DataFrame([{
        "HomeTeam": "Go Ahead Eagles", "AwayTeam": "Willem II",
        "HomeWin_Prob": 0.439, "Draw_Prob": 0.327, "AwayWin_Prob": 0.234,
    }]))
    assert display.loc[0, "Home team"] == "▲ Go Ahead Eagles"
    assert display.loc[0, "Away team"] == "Willem II"


def test_team_match_log_uses_selected_team_perspective_and_recent_first():
    matches = pd.DataFrame([
        {
            "MatchDate": "2026-01-01", "HomeTeam": "Ajax", "AwayTeam": "Twente",
            "FullTimeHomeGoals": 2, "FullTimeAwayGoals": 1, "FullTimeResult": "H",
            "HomeShots": 12, "HomeShotsOnTarget": 6, "HomeCorners": 5,
            "HomeFouls": 8, "HomeYellowCards": 1, "HomeRedCards": 0,
            "HomexG_Avg_L5": 1.8, "HomeTeamPointsLast5": 10,
            "HomeGoalDiff_Avg_L5": 0.8, "HomeRestDays": 7,
        },
        {
            "MatchDate": "2026-01-08", "HomeTeam": "PSV", "AwayTeam": "Ajax",
            "FullTimeHomeGoals": 3, "FullTimeAwayGoals": 1, "FullTimeResult": "H",
            "AwayShots": 9, "AwayShotsOnTarget": 3, "AwayCorners": 4,
            "AwayFouls": 11, "AwayYellowCards": 2, "AwayRedCards": 0,
            "AwayxG_Avg_L5": 1.4, "AwayTeamPointsLast5": 7,
            "AwayGoalDiff_Avg_L5": 0.2, "AwayRestDays": 6,
        },
    ])

    log = _team_match_log(matches, "Ajax")

    assert log["Opponent"].tolist() == ["PSV", "Twente"]
    assert log["Venue"].tolist() == ["Away", "Home"]
    assert log["Result"].tolist() == ["L", "W"]
    assert log["Score"].tolist() == ["1–3", "2–1"]
    assert log["Shots"].tolist() == [9, 12]


def test_team_deep_dive_dictionary_is_unique_and_covers_curated_metrics():
    dictionary = _team_deep_dive_dictionary()

    assert dictionary["Field"].is_unique
    assert {"Score", "Rolling xG (L5)", "Venue form points (L5)", "Yellow cards"}.issubset(
        set(dictionary["Field"])
    )
