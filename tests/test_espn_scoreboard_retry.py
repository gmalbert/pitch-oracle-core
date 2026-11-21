from datetime import datetime, timezone

import pytest
import requests

import fetch_upcoming_fixtures as fixture_fetcher
from pitch_oracle_core.leagues import get_league_config


@pytest.mark.parametrize("league, range_rejected", [("belgium", True), ("epl", False)])
def test_scoreboard_retries_tls_failure_on_the_same_dates(monkeypatch, league, range_rejected):
    attempts = []
    delays = []

    def get(url, *, params, timeout):
        dates = params["dates"]
        attempts.append(dates)
        response = requests.Response()
        if range_rejected and "-" in dates:
            response.status_code = 400
            return response
        if attempts.count(dates) == 1 and (not range_rejected or dates == "20261122"):
            raise requests.exceptions.SSLError("TLS handshake failed")
        response.status_code = 200
        response._content = b'{"events": [{"id": "fixture"}]}'
        return response

    monkeypatch.setattr(fixture_fetcher.requests, "get", get)
    monkeypatch.setattr(fixture_fetcher, "sleep", delays.append)

    date_range = "20261121-20261122"
    now = datetime(2026, 11, 21, tzinfo=timezone.utc)
    events = fixture_fetcher._scoreboard_events(get_league_config(league), now, 1, 0.5)

    if range_rejected:
        assert attempts == [date_range, "20261121", "20261122", "20261122"]
        assert delays == [0.5, 1]
        assert events == [{"id": "fixture"}, {"id": "fixture"}]
    else:
        assert attempts == [date_range, date_range]
        assert delays == [1]
        assert events == [{"id": "fixture"}]


def test_persistent_tls_failure_preserves_existing_fixtures(monkeypatch, tmp_path):
    attempts = []
    delays = []

    def get(url, *, params, timeout):
        attempts.append(params)
        raise requests.exceptions.SSLError("TLS handshake failed")

    monkeypatch.setattr(fixture_fetcher.requests, "get", get)
    monkeypatch.setattr(fixture_fetcher, "sleep", delays.append)
    output = tmp_path / "upcoming_fixtures.csv"
    output.write_text("existing fixtures", encoding="utf-8")

    with pytest.raises(requests.exceptions.SSLError, match="TLS handshake failed"):
        fixture_fetcher.fetch_upcoming_fixtures("belgium", output_dir=tmp_path)

    assert len(attempts) == 3
    assert attempts[0] == attempts[1] == attempts[2]
    assert delays == [1, 2]
    assert output.read_text(encoding="utf-8") == "existing fixtures"
