"""Mutation sentinels prove cutoff protection through the actual feature builder."""

import pandas as pd
import pytest

from pitch_oracle_core.features.advanced_team import LedgerMetric, build_advanced_team_features, assert_point_in_time_features


def fixtures():
    return pd.DataFrame([
        {"fixture_id": f"fx:{day}", "kickoff_utc": f"2026-09-{day:02d}T18:00:00Z", "home_team_id": "tm:x", "away_team_id": "tm:y"}
        for day in (1, 8, 15)
    ])


@pytest.mark.parametrize("metric", ["xt_total", "xg", "xgot", "ppda", "field_tilt"])
def test_same_fixture_giant_value_cannot_change_its_prematch_feature(metric):
    source = fixtures()
    rows = pd.DataFrame([
        {"fixture_id": f"fx:{day}", "team_id": "tm:x", "observed_at": f"2026-09-{day:02d}T20:10:00Z", metric: value}
        for day, value in ((1, 1), (8, 1000), (15, 2))
    ])
    registry = (LedgerMetric(metric, metric),)
    result = build_advanced_team_features(source, rows, source, metrics=registry).set_index("fixture_id")
    assert result.loc["fx:8", f"home_{metric}_ewm5"] == 1
    assert result.loc["fx:15", f"home_{metric}_ewm5"] > 1


def test_late_previous_match_observation_is_not_available_at_early_forecast():
    source = fixtures()
    rows = pd.DataFrame([{"fixture_id": "fx:1", "team_id": "tm:x", "observed_at": "2026-09-08T12:00:00Z", "xt_total": 1000}])
    early = build_advanced_team_features(source.iloc[[1]], rows, source, as_of=pd.Timestamp("2026-09-07T18:00:00Z"))
    lineup = build_advanced_team_features(source.iloc[[1]], rows, source, as_of=pd.Timestamp("2026-09-08T17:00:00Z"))
    assert pd.isna(early.iloc[0].home_xt_for_ewm5)
    assert lineup.iloc[0].home_xt_for_ewm5 == 1000


def test_correction_does_not_rewrite_old_forecast_state():
    source = fixtures()
    rows = pd.DataFrame([
        {"fixture_id": "fx:1", "team_id": "tm:x", "observed_at": "2026-09-01T20:10:00Z", "xt_total": 1},
        {"fixture_id": "fx:1", "team_id": "tm:x", "observed_at": "2026-09-08T17:30:00Z", "xt_total": 500},
    ])
    result = build_advanced_team_features(source.iloc[[1]], rows, source, as_of=pd.Timestamp("2026-09-08T17:00:00Z"))
    assert result.iloc[0].home_xt_for_ewm5 == 1
    assert result.iloc[0].feature_observed_at == pd.Timestamp("2026-09-01T20:10:00Z").isoformat()


def test_team_perspective_and_neutral_names_are_registered_consistently():
    source = fixtures()
    rows = pd.DataFrame([
        {"fixture_id": "fx:1", "team_id": "tm:x", "observed_at": "2026-09-01T20:10:00Z", "xt_total": 1, "ppda": 5, "vaep_offensive": 2, "passes_per_sequence": 3},
        {"fixture_id": "fx:1", "team_id": "tm:y", "observed_at": "2026-09-01T20:10:00Z", "xt_total": 4, "ppda": 8, "vaep_offensive": 6, "passes_per_sequence": 7},
    ])
    result = build_advanced_team_features(source.iloc[[1]], rows, source)
    assert result.iloc[0].home_xt_against_ewm5 == 4
    assert result.iloc[0].home_ppda_ewm5 == 5
    assert result.iloc[0].away_vaep_offensive_ewm10 == 6
    assert result.iloc[0].home_passes_per_sequence_ewm5 == 3


def test_cutoff_audit_rejects_future_or_unproven_lineage():
    source = fixtures().iloc[[1]]
    result = source.assign(as_of="2026-09-08T17:00:00Z", feature_observed_at="2026-09-08T17:01:00Z", home_xt_for_ewm5=1)
    with pytest.raises(ValueError, match="cutoff"):
        assert_point_in_time_features(result)
    result["feature_observed_at"] = None
    with pytest.raises(ValueError, match="lineage"):
        assert_point_in_time_features(result)


def test_postkickoff_forecast_is_rejected():
    source = fixtures().iloc[[1]]
    with pytest.raises(ValueError, match="predate"):
        build_advanced_team_features(source, pd.DataFrame(), source, as_of=pd.Timestamp("2026-09-08T18:00:00Z"))
