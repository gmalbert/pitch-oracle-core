import numpy as np
import pandas as pd
import pytest

from pitch_oracle_core.context.revisions import forecast_revision_deltas
from pitch_oracle_core.pitchapi.forecast_lifecycle import capture_hourly_forecasts, closing_forecasts, replay_forecast, lineup_signature
from pitch_oracle_core.pitchapi.normalize import normalize_lineups

KICKOFF = pd.Timestamp("2026-10-10T15:00:00Z")


def fixture():
    return pd.DataFrame([{"fixture_id": "fixture", "kickoff_utc": KICKOFF, "home_team_id": "home", "away_team_id": "away", "feature_observed_at": "2026-10-01T12:00:00Z"}])


def snapshot(at, home_confirmed=False, away_confirmed=False, replacement=False):
    def team(confirmed, offset):
        return {"confirmed": confirmed, "formation": "4-3-3", "starters": [{"player_id": str(i + offset + (100 if replacement and i == 0 and offset == 0 else 0)), "name": str(i)} for i in range(11)]}
    return normalize_lineups({"home": team(home_confirmed, 0), "away": team(away_confirmed, 20)}, fixture_id="fixture", match_id="provider", home_team_id="home", away_team_id="away", snapshot_at=at.to_pydatetime(), kickoff_utc=KICKOFF.to_pydatetime())


def predict(_inputs):
    return np.array([[.5, .3, .2]]), pd.DataFrame([{"model_id": "baseline", "model_fingerprint": "contract-sha", "fallback_used": True, "fallback_reason": "no_promoted_model"}])


def capture(path, at, snapshots=pd.DataFrame(), predictor=predict):
    return capture_hourly_forecasts(fixture(), snapshots, as_of=at, destination=path, predictor=predictor)


def test_hourly_stage_lifecycle_keeps_actual_times_and_independent_confirmations(tmp_path):
    path = tmp_path / "ledger.parquet"
    initial = KICKOFF - pd.Timedelta(days=3)
    ledger, errors = capture(path, initial)
    assert not errors
    assert ledger.revision_label.tolist() == ["initial"]
    at24 = KICKOFF - pd.Timedelta(hours=23, minutes=40)
    ledger, errors = capture(path, at24, snapshot(at24))
    assert not errors
    assert ledger.revision_label.tolist() == ["initial", "24_hour"]
    home_at = KICKOFF - pd.Timedelta(hours=2)
    archive = pd.concat([snapshot(at24), snapshot(home_at, home_confirmed=True)])
    ledger, errors = capture(path, home_at, archive)
    assert not errors
    assert ledger.iloc[-1].revision_label == "lineup"
    assert ledger.iloc[-1].home_lineup_status == "confirmed"
    assert ledger.iloc[-1].away_lineup_status == "predicted"
    away_at = KICKOFF - pd.Timedelta(hours=1)
    archive = pd.concat([archive, snapshot(away_at, home_confirmed=True, away_confirmed=True)])
    ledger, errors = capture(path, away_at, archive)
    assert not errors
    assert ledger.iloc[-1].revision_label == "lineup"
    final_at = KICKOFF - pd.Timedelta(minutes=10)
    archive = pd.concat([archive, snapshot(final_at, home_confirmed=True, away_confirmed=True)])
    ledger, errors = capture(path, final_at, archive)
    assert not errors
    assert ledger.iloc[-1].revision_label == "hourly"
    closing = closing_forecasts(ledger, as_of=KICKOFF + pd.Timedelta(minutes=5))
    assert len(closing) == 1
    assert closing.iloc[0].issued_at == final_at
    assert closing.iloc[0].revision_label == "closing"
    assert len(ledger) == 5
    assert replay_forecast(ledger, fixture_id="fixture", as_of=home_at).away_lineup_status == "predicted"
    assert replay_forecast(ledger, fixture_id="fixture", as_of=initial - pd.Timedelta(minutes=1)) is None


def test_unchanged_snapshot_time_does_not_trigger_new_lineup_event_but_correction_does(tmp_path):
    path = tmp_path / "ledger.parquet"
    initial = KICKOFF - pd.Timedelta(hours=3)
    first = snapshot(initial, home_confirmed=True)
    capture(path, initial, first)
    at = initial + pd.Timedelta(hours=1)
    unchanged = snapshot(at, home_confirmed=True)
    assert lineup_signature(first) == lineup_signature(unchanged)
    ledger, errors = capture(path, at, pd.concat([first, unchanged]))
    assert not errors
    assert ledger.iloc[-1].revision_label == "24_hour"
    correction_at = at + pd.Timedelta(hours=1)
    corrected = snapshot(correction_at, home_confirmed=True, replacement=True)
    ledger, errors = capture(path, correction_at, pd.concat([first, unchanged, corrected]))
    assert not errors
    assert ledger.iloc[-1].revision_label == "lineup"


def test_failed_hourly_inference_preserves_last_successful_closing(tmp_path):
    path = tmp_path / "ledger.parquet"
    at = KICKOFF - pd.Timedelta(hours=2)
    first, _ = capture(path, at)
    def failed(_inputs):
        raise ValueError("Model unavailable")
    ledger, errors = capture(path, KICKOFF - pd.Timedelta(minutes=10), predictor=failed)
    assert len(errors) == 1
    pd.testing.assert_frame_equal(first, ledger, check_dtype=False)
    assert closing_forecasts(ledger, as_of=KICKOFF).iloc[0].issued_at == at
    assert closing_forecasts(ledger, as_of=KICKOFF - pd.Timedelta(minutes=1)).empty


def test_identical_refresh_is_idempotent_and_late_issue_is_skipped(tmp_path):
    path = tmp_path / "ledger.parquet"
    at = KICKOFF - pd.Timedelta(hours=2)
    first, errors = capture(path, at)
    second, errors = capture(path, at)
    assert not errors
    assert len(second) == 1
    pd.testing.assert_frame_equal(first, second, check_dtype=False)
    late, errors = capture(path, KICKOFF)
    assert not errors
    assert len(late) == 1


def test_revision_deltas_preserve_event_labels_even_when_clock_would_say_closing(tmp_path):
    path = tmp_path / "ledger.parquet"
    capture(path, KICKOFF - pd.Timedelta(hours=2))
    ledger, _ = capture(path, KICKOFF - pd.Timedelta(minutes=30), snapshot(KICKOFF - pd.Timedelta(minutes=30), home_confirmed=True))
    assert forecast_revision_deltas(ledger).iloc[-1].revision_label == "lineup"
    with pytest.raises(ValueError, match="Unknown explicit"):
        forecast_revision_deltas(ledger.assign(revision_label="invented"))


def test_old_schedule_forecast_does_not_become_close_for_rescheduled_fixture(tmp_path):
    path = tmp_path / "ledger.parquet"
    ledger, _ = capture(path, KICKOFF - pd.Timedelta(hours=2))
    current = fixture().assign(kickoff_utc=KICKOFF + pd.Timedelta(days=1))
    assert closing_forecasts(ledger, as_of=KICKOFF + pd.Timedelta(minutes=5), current_fixtures=current).empty
