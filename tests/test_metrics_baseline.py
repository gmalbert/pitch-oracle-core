"""Tests for pitch_oracle_core.evaluation.baseline — proper scores and calibration."""

import numpy as np
import pandas as pd
import pytest

from pitch_oracle_core.evaluation.baseline import (
    calibration_error_summary,
    proper_score_summary,
    reliability_table,
)


class TestProperScoreSummary:
    def test_perfect_predictions_have_zero_scores(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        p = np.array([[1, 0, 0]] * 2 + [[0, 1, 0]] * 2 + [[0, 0, 1]] * 2)
        scores = proper_score_summary(y, p)
        assert scores["brier"] == pytest.approx(0.0, abs=1e-10)
        assert scores["log_loss"] == pytest.approx(0.0, abs=1e-10)
        assert scores["rps"] == pytest.approx(0.0, abs=1e-10)
        assert scores["fixtures"] == 6

    def test_uniform_predictions_have_nonzero_scores(self):
        y = np.array([0, 1, 2])
        p = np.ones((3, 3)) / 3
        scores = proper_score_summary(y, p)
        assert scores["brier"] > 0
        assert scores["log_loss"] > 0
        assert scores["rps"] > 0

    def test_invalid_outcome_raises(self):
        with pytest.raises(ValueError, match="outcomes"):
            proper_score_summary(np.array([3]), np.array([[0.5, 0.3, 0.2]]))

    def test_invalid_probabilities_raise(self):
        with pytest.raises(ValueError, match="probabilities"):
            proper_score_summary(np.array([0]), np.array([[0.5, 0.3]]))


class TestReliabilityTable:
    def test_shape_matches_n_bins(self):
        p = np.linspace(0.05, 0.95, 20)
        y = (p > 0.5).astype(float)
        table = reliability_table(p, y, n_bins=5)
        assert len(table) == 5

    def test_columns_present(self):
        p = np.array([0.1, 0.5, 0.9])
        y = np.array([0, 1, 1])
        table = reliability_table(p, y, n_bins=3)
        assert set(table.columns) == {"bin_midpoint", "n", "fraction_positive", "predicted", "gap"}

    def test_empty_bin_has_nan(self):
        p = np.array([0.1, 0.2])
        y = np.array([0, 1])
        table = reliability_table(p, y, n_bins=10)
        empty = table[table["n"] == 0]
        assert empty["fraction_positive"].isna().all()

    def test_n_bins_must_be_at_least_two(self):
        with pytest.raises(ValueError, match="n_bins"):
            reliability_table(np.array([0.5]), np.array([1]), n_bins=1)


class TestCalibrationErrorSummary:
    def test_perfect_calibration_has_zero_ece(self):
        p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = calibration_error_summary(p, y, n_bins=5)
        assert result["ece"] >= 0
        assert result["mce"] >= 0

    def test_returns_ece_and_mce(self):
        p = np.array([0.3, 0.7])
        y = np.array([0, 1])
        result = calibration_error_summary(p, y, n_bins=2)
        assert "ece" in result
        assert "mce" in result
        assert result["mce"] >= result["ece"]
