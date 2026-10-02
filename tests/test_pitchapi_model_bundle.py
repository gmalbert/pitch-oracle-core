from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pitch_oracle_core.features.families import FeatureFamilyConfig
from pitch_oracle_core.features.forecast_inputs import build_forecast_inputs
from pitch_oracle_core.leagues import get_league_config
from pitch_oracle_core.pitchapi.model_bundle import fit_bundle, load_bundle, train_model_bundles, predict_with_fallback


def training():
    dates = pd.date_range("2025-08-01", periods=90, freq="D")
    return pd.DataFrame({"MatchDate": dates.strftime("%Y-%m-%d"), "HomeTeam": "Ajax", "AwayTeam": "PSV", "FullTimeResult": np.tile(["H", "D", "A"], 30), "FullTimeHomeGoals": np.tile([2, 1, 0], 30), "FullTimeAwayGoals": np.tile([0, 1, 2], 30), "EloDiff": np.tile([30, 0, -20], 30), "home_xt_for_ewm10": np.tile([2, 1, .5], 30), "home_xg": np.tile([9, 8, 7], 30), "provider_schema_version": 2})


def live(bundle):
    cutoff = pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=1)
    return pd.DataFrame([{"fixture_id": "upcoming", "HomeTeam": "Ajax", "AwayTeam": "PSV", "kickoff_utc": cutoff + pd.Timedelta(days=1), "as_of": cutoff, "EloDiff": 15, "home_xt_for_ewm10": 1.5, "feature_observed_at": cutoff - pd.Timedelta(hours=1)}]), cutoff


def test_models_have_independent_widths_roundtrip_and_baseline_fallback(tmp_path):
    train_model_bundles(training(), configuration=FeatureFamilyConfig("eredivisie", ("advanced_team",), "validated-run"), models_dir=tmp_path)
    baseline = load_bundle(tmp_path / "pitchapi_baseline.pkl", league_key="eredivisie")
    promoted = load_bundle(tmp_path / "pitchapi_promoted.pkl", league_key="eredivisie")
    assert baseline.contract.feature_names == ("EloDiff",)
    assert promoted.contract.feature_names == ("EloDiff", "home_xt_for_ewm10")
    assert baseline.model.n_features_in_ == 1
    assert promoted.model.n_features_in_ == 2
    upcoming, cutoff = live(baseline)
    probabilities, metadata = predict_with_fallback(training(), upcoming, baseline=baseline, promoted=promoted, capability_health={"checked_at": cutoff.isoformat(), "mapping": {"gate_passed": True}, "capabilities": {"advanced_team": {"status": "stale"}}}, as_of=cutoff)
    assert metadata.iloc[0].fallback_reason == "advanced_team_stale"
    assert metadata.iloc[0].model_id == baseline.model_id
    np.testing.assert_allclose(probabilities, baseline.model.predict_proba(np.array([[15]], dtype=np.float32)))
    probabilities, metadata = predict_with_fallback(training(), upcoming, baseline=baseline, promoted=promoted, capability_health={"checked_at": cutoff.isoformat(), "mapping": {"gate_passed": True}, "capabilities": {"advanced_team": {"status": "available"}}}, as_of=cutoff)
    assert metadata.iloc[0].model_id == promoted.model_id
    assert not metadata.iloc[0].fallback_used
    np.testing.assert_allclose(probabilities, promoted.model.predict_proba(np.array([[15, 1.5]], dtype=np.float32)))


def test_missing_provider_values_use_fitted_baseline_not_column_dropping():
    baseline = fit_bundle(training(), league_key="eredivisie")
    promoted = fit_bundle(training(), league_key="eredivisie", families=("advanced_team",), evidence_id="run", model_id="promoted")
    upcoming, cutoff = live(baseline)
    upcoming["home_xt_for_ewm10"] = np.nan
    _, metadata = predict_with_fallback(training(), upcoming, baseline=baseline, promoted=promoted, capability_health={"checked_at": cutoff.isoformat(), "mapping": {"gate_passed": True}, "capabilities": {"advanced_team": {"status": "available"}}}, as_of=cutoff)
    assert metadata.iloc[0].fallback_reason == "advanced_team_missing_required_features"
    assert metadata.iloc[0].model_id == baseline.model_id


def test_model_fingerprint_and_league_guard():
    baseline = fit_bundle(training(), league_key="eredivisie")
    with pytest.raises(ValueError, match="fingerprint"):
        replace(baseline, fingerprint="wrong").validate(league_key="eredivisie")
    with pytest.raises(ValueError, match="league"):
        baseline.validate(league_key="epl")
    upcoming, cutoff = live(baseline)
    with pytest.raises(ValueError, match="not available"):
        predict_with_fallback(training(), upcoming, baseline=baseline, as_of=pd.Timestamp(baseline.trained_at) - pd.Timedelta(days=1))


def test_late_feature_lineage_selects_baseline():
    baseline = fit_bundle(training(), league_key="eredivisie")
    promoted = fit_bundle(training(), league_key="eredivisie", families=("advanced_team",), evidence_id="run")
    upcoming, cutoff = live(baseline)
    upcoming["feature_observed_at"] = cutoff + pd.Timedelta(hours=1)
    _, metadata = predict_with_fallback(training(), upcoming, baseline=baseline, promoted=promoted, capability_health={"checked_at": cutoff.isoformat(), "mapping": {"gate_passed": True}, "capabilities": {"advanced_team": {"status": "available"}}}, as_of=cutoff)
    assert metadata.iloc[0].fallback_reason == "missing_or_late_feature_lineage"


def test_live_football_state_includes_latest_completed_game_and_excludes_future():
    history = pd.DataFrame([
        {"MatchDate": "2026-08-01", "KickoffTime": "15:00", "HomeTeam": "Ajax", "AwayTeam": "PSV", "FullTimeResult": "H", "FullTimeHomeGoals": 2, "FullTimeAwayGoals": 0},
        {"MatchDate": "2026-08-09", "KickoffTime": "15:00", "HomeTeam": "Ajax", "AwayTeam": "PSV", "FullTimeResult": "A", "FullTimeHomeGoals": 0, "FullTimeAwayGoals": 99},
    ])
    upcoming = pd.DataFrame([{"MatchDate": "2026-08-10", "KickoffTime": "15:00", "HomeTeam": "Ajax", "AwayTeam": "PSV"}])
    result = build_forecast_inputs(history, upcoming, config=get_league_config("eredivisie"), as_of=pd.Timestamp("2026-08-08T12:00:00Z"))
    assert result.iloc[0].home_points_l5 == 3
    assert result.iloc[0].home_goals_for_l5 == 2
    assert result.iloc[0].away_goals_for_l5 == 0


def test_family_without_eligible_training_history_cannot_be_promoted():
    with pytest.raises(ValueError, match="lacks eligible"):
        fit_bundle(training().assign(home_xt_for_ewm10=np.nan), league_key="eredivisie", families=("advanced_team",), evidence_id="run")


def test_mapping_gate_and_stale_audit_force_independent_baseline():
    baseline = fit_bundle(training(), league_key="eredivisie")
    promoted = fit_bundle(training(), league_key="eredivisie", families=("advanced_team",), evidence_id="run")
    upcoming, cutoff = live(baseline)
    health = {"checked_at": cutoff.isoformat(), "mapping": {"gate_passed": False}, "capabilities": {"advanced_team": {"status": "available"}}}
    _, metadata = predict_with_fallback(training(), upcoming, baseline=baseline, promoted=promoted, capability_health=health, as_of=cutoff)
    assert metadata.iloc[0].fallback_reason == "fixture_mapping_not_validated"
    health.update(checked_at=(cutoff - pd.Timedelta(days=2)).isoformat(), mapping={"gate_passed": True})
    _, metadata = predict_with_fallback(training(), upcoming, baseline=baseline, promoted=promoted, capability_health=health, as_of=cutoff)
    assert metadata.iloc[0].fallback_reason == "capability_audit_stale_or_missing"
