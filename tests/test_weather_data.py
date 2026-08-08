import pandas as pd

import pitch_oracle_core.weather as weather


def test_weather_provider_failure_stops_after_first_request(monkeypatch, tmp_path):
    calls = []

    def unavailable(stadium, match_date, *, stadium_coords=None, raise_on_error=False, timezone=""):
        calls.append((stadium, match_date, raise_on_error))
        raise RuntimeError("offline")

    monkeypatch.setattr(weather, "fetch_match_weather", unavailable)
    matches = pd.DataFrame(
        [
            {"HomeTeam": "Arsenal", "MatchDate": "2026-01-01"},
            {"HomeTeam": "Chelsea", "MatchDate": "2026-01-02"},
        ]
    )

    result = weather.add_weather_features(
        matches,
        stadium_map={"Arsenal": "Arsenal", "Chelsea": "Chelsea"},
        stadium_coords={
            "Arsenal": {"lat": 51.5549, "lon": -0.1084},
            "Chelsea": {"lat": 51.4817, "lon": -0.1910},
        },
        data_dir=str(tmp_path),
        fetcher=unavailable,
    )

    assert len(calls) == 1
    assert calls[0][2] is True
    assert result["WeatherCondition"].tolist() == ["Unknown", "Unknown"]
    assert result["Precipitation"].tolist() == [0, 0]


def test_weather_requests_use_the_configured_timezone_and_cache_file(monkeypatch, tmp_path):
    calls = []

    def available(stadium, match_date, *, stadium_coords=None, raise_on_error=False, timezone=""):
        calls.append((stadium, timezone))
        return {
            "Temperature": 10, "Humidity": 70, "WindSpeed": 3,
            "Precipitation": 0, "WeatherCondition": "Clear sky",
            "WeatherDescription": "Clear sky",
        }

    matches = pd.DataFrame([{"HomeTeam": "Ajax", "MatchDate": "2026-01-01"}])
    weather.add_weather_features(
        matches,
        stadium_map={"Ajax": "Ajax"},
        stadium_coords={"Ajax": {"lat": 52.3140, "lon": 4.9414}},
        cache_file="weather_cache_eredivisie.csv",
        data_dir=str(tmp_path),
        timezone="Europe/Amsterdam",
        fetcher=available,
    )

    assert calls == [("Ajax", "Europe/Amsterdam")]
    assert (tmp_path / "weather_cache_eredivisie.csv").is_file()


def test_weather_accepts_upcoming_fixture_date_column(monkeypatch, tmp_path):
    calls = []

    def available(stadium, match_date, *, stadium_coords=None, raise_on_error=False, timezone=""):
        calls.append((stadium, match_date))
        return {
            "Temperature": 18, "Humidity": 65, "WindSpeed": 4,
            "Precipitation": 0.3, "WeatherCondition": "Partly cloudy",
            "WeatherDescription": "Partly cloudy",
        }

    fixtures = pd.DataFrame([{"HomeTeam": "Ajax", "Date": "2099-08-08"}])
    result = weather.add_weather_features(
        fixtures,
        stadium_map={"Ajax": "Ajax"},
        stadium_coords={"Ajax": {"lat": 52.3140, "lon": 4.9414}},
        data_dir=str(tmp_path),
        fetcher=available,
    )

    assert calls == [("Ajax", "2099-08-08")]
    assert result.loc[0, "WeatherDescription"] == "Partly cloudy"
    assert result.loc[0, "Temperature"] == 18
