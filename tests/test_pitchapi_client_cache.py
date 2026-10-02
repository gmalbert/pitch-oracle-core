"""Transport failures and actual immutable-cache behavior, independent of live API."""

from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock

import pytest
import requests

from pitch_oracle_core.pitchapi.cache import ObservationCache
from pitch_oracle_core.pitchapi.client import PitchAPIClient, PitchAPIError
from pitch_oracle_core.pitchapi.contracts import LineupSnapshotRow, LineupStatus

NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


def response(status=200, data=None, headers=None):
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = {"data": {"matches": []}} if data is None else data
    return result


def client(*responses):
    session = Mock(headers={})
    session.get.side_effect = responses
    return PitchAPIClient("test-secret", session=session, sleep=Mock())


def test_404_analytics_code_is_parsed_before_http_exception():
    transport = client(response(404, {"error": {"code": "ANALYTICS_UNAVAILABLE", "message": "unrated"}}))
    assert transport.advanced_team("m_test") is None
    assert transport.session.get.call_count == 1


def test_not_found_is_optional_only_for_lineups():
    payload = {"error": {"code": "RESOURCE_NOT_FOUND", "message": "missing"}}
    assert client(response(404, payload)).lineups("m_test") is None
    with pytest.raises(PitchAPIError) as caught:
        client(response(404, payload)).advanced_team("m_test")
    assert caught.value.code == "RESOURCE_NOT_FOUND"
    assert caught.value.status_code == 404


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retry_is_bounded_and_recovers(status):
    transport = client(response(status, {"error": {"code": "RETRY", "message": "later"}}, {"Retry-After": "999"}), response())
    assert transport.league_matches("l_test", "2026/2027", status="upcoming") == []
    transport.sleep.assert_called_once_with(30.0)
    assert transport.session.get.call_args.kwargs["params"]["status"] == "upcoming"


def test_429_cannot_recurse_indefinitely():
    transport = client(*[response(429, {"error": {"code": "RATE_LIMIT_EXCEEDED"}}) for _ in range(4)])
    with pytest.raises(PitchAPIError, match="RATE_LIMIT_EXCEEDED"):
        transport.leagues()
    assert transport.session.get.call_count == 4
    assert transport.sleep.call_count == 3


@pytest.mark.parametrize("envelope", [{}, {"data": None}, [], {"data": "bad"}])
def test_malformed_response_fails_without_silent_empty_data(envelope):
    with pytest.raises(PitchAPIError, match="MALFORMED_RESPONSE"):
        client(response(200, envelope)).leagues()


def test_authentication_failure_does_not_retry_or_expose_credentials():
    transport = client(response(401, {"error": {"code": "UNAUTHORIZED", "message": "invalid key"}}))
    with pytest.raises(PitchAPIError) as caught:
        transport.leagues()
    assert transport.session.get.call_count == 1
    assert "test-secret" not in str(caught.value)
    assert "test-secret" not in repr(transport)


def test_network_error_retries_and_reports_transport_code():
    transport = client(*[requests.Timeout("private request") for _ in range(4)])
    with pytest.raises(PitchAPIError, match="TRANSPORT_ERROR"):
        transport.leagues()
    assert transport.session.get.call_count == 4


def test_raw_revision_is_immutable_and_correction_preserves_history(tmp_path):
    cache = ObservationCache(tmp_path)
    cache.store("shots_m_test", "shots", {"xg": 1}, now=NOW)
    original = next(path for path in (tmp_path / "shots_m_test").glob("*.json") if path.name != "latest.json")
    before = original.read_bytes()
    cache.store("shots_m_test", "shots", {"xg": 2}, now=NOW + timedelta(days=1))
    assert original.read_bytes() == before
    assert len(cache.revisions("shots_m_test")) == 2
    assert cache.revisions("shots_m_test", as_of=NOW)[0].payload == {"xg": 1}
    assert cache.latest("shots_m_test").payload == {"xg": 2}


def test_same_payload_advances_check_time_without_fabricating_observation_time(tmp_path):
    cache = ObservationCache(tmp_path)
    cache.store("shots_m_test", "shots", [], now=NOW)
    observation = cache.store("shots_m_test", "shots", [], now=NOW + timedelta(days=1))
    assert observation.observed_at == NOW
    assert observation.checked_at == NOW + timedelta(days=1)
    assert len(cache.revisions("shots_m_test")) == 1


def test_correction_window_and_manual_refresh(tmp_path):
    cache = ObservationCache(tmp_path)
    loader = Mock(return_value={"xg": 1})
    completion = NOW - timedelta(hours=2)
    cache.fetch("shots_m_test", "shots", loader, now=NOW, completed_at=completion)
    cache.fetch("shots_m_test", "shots", loader, now=NOW + timedelta(hours=1), completed_at=completion)
    assert loader.call_count == 1
    cache.fetch("shots_m_test", "shots", loader, now=NOW + timedelta(days=1), completed_at=completion)
    assert loader.call_count == 2
    cache.fetch("shots_m_test", "shots", loader, now=NOW + timedelta(days=8), completed_at=completion)
    assert loader.call_count == 2
    cache.fetch("shots_m_test", "shots", loader, now=NOW + timedelta(days=8), completed_at=completion, refresh=True)
    assert loader.call_count == 3


def test_cache_failure_retains_data_and_reports_failure(tmp_path):
    cache = ObservationCache(tmp_path)
    cache.store("shots_m_test", "shots", {"xg": 1}, now=NOW)
    loader = Mock(side_effect=PitchAPIError("INTERNAL_SERVER_ERROR"))
    cached = cache.fetch("shots_m_test", "shots", loader, now=NOW + timedelta(days=1))
    assert cached.payload == {"xg": 1}
    assert cached.error_code == "INTERNAL_SERVER_ERROR"
    assert cached.observed_at == NOW
    with pytest.raises(PitchAPIError):
        cache.fetch("shots_m_test", "shots", loader, now=NOW + timedelta(days=1), allow_cached_on_error=False)


def test_failed_first_fetch_does_not_create_a_false_zero_cache(tmp_path):
    cache = ObservationCache(tmp_path)
    with pytest.raises(PitchAPIError):
        cache.fetch("shots_m_test", "shots", Mock(side_effect=PitchAPIError("ANALYTICS_UNAVAILABLE")), now=NOW)
    assert cache.latest("shots_m_test") is None


def test_cache_detects_corruption_and_path_escape(tmp_path):
    cache = ObservationCache(tmp_path)
    with pytest.raises(ValueError, match="path-safe"):
        cache.store("../secret", "shots", {}, now=NOW)
    cache.store("shots_m_test", "shots", {"xg": 1}, now=NOW)
    raw = next(path for path in (tmp_path / "shots_m_test").glob("*.json") if path.name != "latest.json")
    payload = json.loads(raw.read_text())
    payload["payload"]["xg"] = 9
    raw.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        cache.latest("shots_m_test")


def test_postkickoff_lineup_is_stored_but_not_eligible():
    row = LineupSnapshotRow("fx:test", "m_test", "tm:test", "p_test", "Player", None, True, LineupStatus.CONFIRMED, NOW, NOW)
    assert not row.prematch_eligible
