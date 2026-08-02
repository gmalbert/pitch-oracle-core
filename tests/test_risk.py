import pytest

from pitch_oracle_core.risk import (
    calculate_prediction_risk,
    get_prediction_guidance,
    get_risk_category,
)


def test_risk_scale_has_meaningful_endpoints():
    uniform_risk, uniform_confidence = calculate_prediction_risk([1 / 3, 1 / 3, 1 / 3])
    certain_risk, certain_confidence = calculate_prediction_risk([1, 0, 0])

    assert uniform_risk == pytest.approx(100)
    assert uniform_confidence == pytest.approx(0)
    assert certain_risk == pytest.approx(0)
    assert certain_confidence == pytest.approx(1)


def test_risk_falls_as_forecast_becomes_more_decisive():
    close, _ = calculate_prediction_risk([0.39, 0.37, 0.24])
    useful, _ = calculate_prediction_risk([0.59, 0.29, 0.12])
    strong, _ = calculate_prediction_risk([0.76, 0.13, 0.11])

    assert strong < useful < close
    assert get_risk_category(strong)[0] == "Low Risk"
    assert get_risk_category(useful)[0] == "Moderate Risk"
    assert get_risk_category(close)[0] == "Critical Risk"


def test_risk_normalizes_rounded_probabilities():
    exact = calculate_prediction_risk([0.68, 0.23, 0.09])
    rounded_percent = calculate_prediction_risk([68, 23, 9])
    assert rounded_percent == pytest.approx(exact)


def test_guidance_distinguishes_a_lean_from_an_ambiguous_match():
    assert get_prediction_guidance([0.76, 0.13, 0.11], 35) == ("Strong Home Lean", "✅")
    assert get_prediction_guidance([0.59, 0.29, 0.12], 70) == ("Consider Home", "🤔")
    assert get_prediction_guidance([0.39, 0.37, 0.24], 96) == ("No Clear Edge", "⏸️")


@pytest.mark.parametrize("probabilities", ([0, 0, 0], [-0.1, 0.5, 0.6], [0.5, 0.5]))
def test_invalid_probabilities_are_rejected(probabilities):
    with pytest.raises(ValueError):
        calculate_prediction_risk(probabilities)
