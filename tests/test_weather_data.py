import pandas as pd

import fetch_weather_data as weather


def test_weather_provider_failure_stops_after_first_request(monkeypatch, tmp_path):
    calls = []

    def unavailable(stadium, match_date, api_key=None, *, raise_on_error=False, timezone=""):
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


def test_weather_requests_use_the_configured_timezone_and_cache_file(monkeypatch, tmp_path):
    calls = []

    def available(stadium, match_date, api_key=None, *, raise_on_error=False, timezone=""):
        calls.append((stadium, timezone))
        return {
            "Temperature": 10, "Humidity": 70, "WindSpeed": 3,
            "Precipitation": 0, "WeatherCondition": "Clear sky",
            "WeatherDescription": "Clear sky",
        }

    monkeypatch.setattr(weather, "fetch_match_weather", available)
    matches = pd.DataFrame([{"HomeTeam": "Ajax", "MatchDate": "2026-01-01"}])
    weather.add_weather_features(
        matches,
        stadium_map={"Ajax": "Ajax"},
        stadium_coords={"Ajax": {"lat": 52.3140, "lon": 4.9414}},
        cache_file="weather_cache_eredivisie.csv",
        data_dir=str(tmp_path),
        timezone="Europe/Amsterdam",
    )

    assert calls == [("Ajax", "Europe/Amsterdam")]
    assert (tmp_path / "weather_cache_eredivisie.csv").is_file()
