import pytest

from pitch_oracle_core.best_bets import market_metrics


def test_market_metrics_remove_overround_and_compute_expected_value():
    market_probability, edge, expected_value = market_metrics(0.55, 2.0, [2.0, 3.5, 4.0])
    assert market_probability == pytest.approx(0.5 / (0.5 + 1 / 3.5 + 0.25))
    assert edge == pytest.approx(0.55 - market_probability)
    assert expected_value == pytest.approx(0.10)
