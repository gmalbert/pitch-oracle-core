import pandas as pd

from pitch_oracle_core.features.families import FeatureFamilyConfig
from pitch_oracle_core.pitchapi.artifacts import publish_index
from pitch_oracle_core.pitchapi.forecast_lifecycle import current_prediction_projection
from pitch_oracle_core.pitchapi.serving import legacy_prediction_overlay
from pitch_oracle_core.pitchapi.storage import write_frame


def test_legacy_overlay_uses_actual_recent_issue_and_preserves_history_and_alternate_models(tmp_path):
    now = pd.Timestamp("2026-10-02T12:00:00Z")
    fixtures = pd.DataFrame([{"fixture_id": "f", "kickoff_utc": now + pd.Timedelta(days=1), "MatchDate": "2026-10-03", "HomeTeam": "Home", "AwayTeam": "Away"}])
    ledger = pd.DataFrame([{"fixture_id": "f", "kickoff_utc": fixtures.iloc[0].kickoff_utc, "issued_at": now.isoformat(), "model_id": "baseline", "model_fingerprint": "sha", "revision_label": "lineup", "p_home": .6, "p_draw": .25, "p_away": .15, "fallback_used": True, "fallback_reason": "lineup_missing", "feature_families": ""}])
    projection = current_prediction_projection(ledger, fixtures, as_of=now, evidence_id="run")
    data = tmp_path / "data_files"
    data.mkdir()
    write_frame(projection, data / "pitchapi_upcoming_predictions.csv")
    config = FeatureFamilyConfig("epl", ("confirmed_lineup",), "run")
    (data / "pitchapi_feature_config.json").write_text(__import__("json").dumps(config.as_dict()))
    publish_index(tmp_path, "epl")
    log = pd.DataFrame([{"MatchDate": "2026-10-03", "HomeTeam": "Home", "AwayTeam": "Away", "ActualResult": None, "ModelVersion": version, "PredHomeWin": 50} for version in ("ensemble_v1", "nn_v1")] + [{"MatchDate": "2026-09-01", "HomeTeam": "Home", "AwayTeam": "Away", "ActualResult": "H", "ModelVersion": "ensemble_v1", "PredHomeWin": 40}])
    current = legacy_prediction_overlay(log, league_key="epl", root=tmp_path, now=now)
    assert len(current) == 3
    assert current.loc[current.ModelVersion.eq("pitchapi_v1")].iloc[0].PredHomeWin == 60
    assert current.loc[current.ActualResult.eq("H")].iloc[0].PredHomeWin == 40
    pd.testing.assert_frame_equal(legacy_prediction_overlay(log, league_key="epl", root=tmp_path, now=now + pd.Timedelta(hours=3)), log)
    # A rescheduled event must await a new issue for its new kickoff.
    assert current_prediction_projection(ledger, fixtures.assign(kickoff_utc=now + pd.Timedelta(days=2)), as_of=now).empty
    # Tampering with the optional projection cannot replace the legacy forecast.
    write_frame(projection.assign(PredHomeWin=99), data / "pitchapi_upcoming_predictions.csv")
    pd.testing.assert_frame_equal(legacy_prediction_overlay(log, league_key="epl", root=tmp_path, now=now), log)
