"""Tests for pitch_oracle_core.implied — penaltyblog-backed de-vig conversions."""

import pytest

from penaltyblog.implied import ImpliedMethod

from pitch_oracle_core.implied import (
    DEFAULT_METHOD,
    FairOdds,
    implied_probabilities,
    no_vig_probability,
)


FIXTURE_ODDS = [2.10, 3.40, 3.60]


class TestImpliedProbabilities:
    @pytest.mark.parametrize("method", list(ImpliedMethod))
    def test_all_seven_methods_sum_to_one(self, method):
        result = implied_probabilities(FIXTURE_ODDS, method=method)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-10

    @pytest.mark.parametrize("method", list(ImpliedMethod))
    def test_all_seven_methods_have_positive_margin(self, method):
        result = implied_probabilities(FIXTURE_ODDS, method=method)
        assert result.margin > 0

    def test_default_method_is_logarithmic(self):
        result = implied_probabilities(FIXTURE_ODDS)
        assert result.method == ImpliedMethod.LOGARITHMIC

    def test_market_names_are_preserved(self):
        result = implied_probabilities(FIXTURE_ODDS, market_names=("home", "draw", "away"))
        assert set(result.probabilities.keys()) == {"home", "draw", "away"}

    def test_getitem_returns_probability(self):
        result = implied_probabilities(FIXTURE_ODDS)
        assert result["home"] == result.probabilities["home"]

    def test_home_is_most_likely(self):
        result = implied_probabilities(FIXTURE_ODDS)
        assert result["home"] > result["draw"]
        assert result["home"] > result["away"]


class TestNoVigProbability:
    def test_returns_probability_and_margin(self):
        prob, margin = no_vig_probability(2.10, FIXTURE_ODDS)
        assert 0 < prob < 1
        assert margin > 0

    def test_matches_implied_probabilities(self):
        fair = implied_probabilities(FIXTURE_ODDS)
        prob, _ = no_vig_probability(2.10, FIXTURE_ODDS)
        assert abs(prob - fair["home"]) < 1e-10
