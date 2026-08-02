import pandas as pd

from pitch_oracle_core.ui_pages import _filter_predictions_by_risk, _style_prediction_risk


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
