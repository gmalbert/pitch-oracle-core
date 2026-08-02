import pandas as pd

import fetch_weather_data as weather


def test_weather_provider_failure_stops_after_first_request(monkeypatch, tmp_path):
    calls = []

    def unavailable(stadium, match_date, api_key=None, *, raise_on_error=False):
        calls.append((stadium, match_date, raise_on_error))
        raise RuntimeError("offline")

    monkeypatch.setattr(weather, "fetch_match_weather", unavailable)
    matches = pd.DataFrame(
        [
            {"HomeTeam": "Arsenal", "MatchDate": "2026-01-01"},
            {"HomeTeam": "Chelsea", "MatchDate": "2026-01-02"},
        ]
    )

    result = weather.add_weather_features(matches, data_dir=str(tmp_path))

    assert len(calls) == 1
    assert calls[0][2] is True
    assert result["WeatherCondition"].tolist() == ["Unknown", "Unknown"]
    assert result["Precipitation"].tolist() == [0, 0]
