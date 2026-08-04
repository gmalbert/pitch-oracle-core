"""Provider-neutral historical match-weather features.

Consumers supply the league's team/stadium aliases and coordinates from
``LeagueConfig``.  This keeps the Open-Meteo integration and cache behavior in
the shared core instead of copying it into every league repository.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Mapping

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


_WEATHER_DESCRIPTIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle",
    53: "Moderate drizzle", 55: "Dense drizzle", 56: "Light freezing drizzle",
    57: "Dense freezing drizzle", 61: "Slight rain", 63: "Moderate rain",
    65: "Heavy rain", 66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
    77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers", 85: "Slight snow showers",
    86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def _client():
    session = requests_cache.CachedSession(".cache", expire_after=-1)
    return openmeteo_requests.Client(session=retry(session, retries=5, backoff_factor=0.2))


def fetch_match_weather(
    stadium_location: str,
    match_date,
    *,
    stadium_coords: Mapping[str, Mapping[str, float]] | None = None,
    raise_on_error: bool = False,
    timezone: str = "Europe/London",
    client=None,
) -> dict | None:
    """Fetch historical weather for a stadium/date from Open-Meteo."""
    date = match_date.strftime("%Y-%m-%d") if hasattr(match_date, "strftime") else str(match_date).split(" ")[0]
    coords = (stadium_coords or {}).get(stadium_location)
    if not coords:
        return None

    try:
        params = {
            "latitude": coords["lat"], "longitude": coords["lon"],
            "start_date": date, "end_date": date,
            "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m", "wind_speed_10m"],
            "daily": ["weathercode"], "temperature_unit": "celsius",
            "wind_speed_unit": "ms", "timezone": timezone,
        }
        response = (client or _client()).weather_api(
            "https://archive-api.open-meteo.com/v1/archive", params=params
        )[0]
        hourly = response.Hourly()
        values = [hourly.Variables(i).ValuesAsNumpy() for i in range(4)]
        hours = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left",
        ).hour
        if not len(hours):
            return None
        index = min(range(len(hours)), key=lambda i: abs(hours[i] - 16))
        weather_code = int(response.Daily().Variables(0).ValuesAsNumpy()[0])
        condition = _WEATHER_DESCRIPTIONS.get(weather_code, "Unknown")
        return {
            "Temperature": float(values[0][index]), "Humidity": float(values[2][index]),
            "WindSpeed": float(values[3][index]), "Precipitation": float(values[1][index]),
            "WeatherCondition": condition, "WeatherDescription": condition,
        }
    except Exception as exc:
        if raise_on_error:
            raise RuntimeError(f"weather request failed for {stadium_location} on {date}") from exc
        return None


def _empty_features(size: int) -> pd.DataFrame:
    return pd.DataFrame({
        "Temperature": [None] * size, "Humidity": [None] * size,
        "WindSpeed": [None] * size, "Precipitation": [0] * size,
        "WeatherCondition": ["Unknown"] * size,
        "WeatherDescription": ["Unknown"] * size,
    })


def add_weather_features(
    df: pd.DataFrame,
    *,
    cache_file: str = "weather_cache.csv",
    stadium_map: Mapping[str, str] | None = None,
    stadium_coords: Mapping[str, Mapping[str, float]] | None = None,
    data_dir: str | os.PathLike[str] = "data_files",
    timezone: str = "Europe/London",
    fetcher=None,
) -> pd.DataFrame:
    """Add cached historical weather features using league-supplied coordinates."""
    cache_path = Path(data_dir) / cache_file
    try:
        cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    except Exception:
        cached = pd.DataFrame()
    cached_keys = set(cached.get("cache_key", pd.Series(dtype=str)).dropna())
    requests = []
    for _, match in df.iterrows():
        stadium = (stadium_map or {}).get(match.get("HomeTeam"))
        if pd.notna(stadium):
            date = pd.Timestamp(match["MatchDate"]).strftime("%Y-%m-%d")
            key = f"{match['HomeTeam']}_{date}"
            if key not in cached_keys:
                requests.append((stadium, match["MatchDate"], key))

    fetcher = fetcher or fetch_match_weather
    new_rows = []
    for stadium, date, key in requests:
        try:
            weather = fetcher(
                stadium, date, stadium_coords=stadium_coords,
                raise_on_error=True, timezone=timezone,
            )
        except Exception:
            # A provider outage must not prevent model-data preparation. Any
            # existing cache remains usable and missing rows get safe defaults.
            break
        if weather:
            new_rows.append({**weather, "cache_key": key, "stadium": stadium})
    if new_rows:
        combined = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates("cache_key")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cache_path, index=False)
        cached = combined

    if cached.empty:
        return pd.concat([df.reset_index(drop=True), _empty_features(len(df))], axis=1)
    features = []
    for _, match in df.iterrows():
        key = f"{match['HomeTeam']}_{pd.Timestamp(match['MatchDate']).strftime('%Y-%m-%d')}"
        rows = cached[cached["cache_key"] == key]
        if rows.empty:
            features.append(_empty_features(1).iloc[0].to_dict())
        else:
            features.append(rows.iloc[0].drop(labels=["cache_key", "stadium"], errors="ignore").to_dict())
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(features)], axis=1)


def categorize_weather_impact(row) -> str:
    if pd.isna(row.get("Precipitation", 0)):
        return "Unknown"
    if row["Precipitation"] > 5:
        return "Heavy Rain"
    if pd.notna(row.get("WindSpeed")) and row["WindSpeed"] > 15:
        return "Windy"
    if pd.notna(row.get("Temperature")) and row["Temperature"] < 5:
        return "Cold"
    if pd.notna(row.get("Temperature")) and row["Temperature"] > 25:
        return "Hot"
    return "Normal"


def add_weather_impact_category(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["WeatherImpact"] = result.apply(categorize_weather_impact, axis=1)
    return result
