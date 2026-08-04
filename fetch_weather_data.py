"""Backward-compatible consumer shim for the shared weather integration."""

from pitch_oracle_core import weather as _weather
from pitch_oracle_core.leagues import get_league_config


_EPL = get_league_config("epl")
STADIUM_MAP = {team: team for team in _EPL.stadium_coordinates}
STADIUM_COORDS = {
    team: {"lat": coords[0], "lon": coords[1]}
    for team, coords in _EPL.stadium_coordinates.items()
}


def fetch_match_weather(stadium_location, match_date, api_key=None, *, raise_on_error=False, timezone="Europe/London"):
    return _weather.fetch_match_weather(
        stadium_location,
        match_date,
        stadium_coords=STADIUM_COORDS,
        raise_on_error=raise_on_error,
        timezone=timezone,
    )


def add_weather_features(df, api_key=None, cache_file="weather_cache.csv", stadium_map=None,
                         stadium_coords=None, data_dir="data_files", timezone="Europe/London"):
    return _weather.add_weather_features(
        df,
        cache_file=cache_file,
        stadium_map=stadium_map or STADIUM_MAP,
        stadium_coords=stadium_coords or STADIUM_COORDS,
        data_dir=data_dir,
        timezone=timezone,
        fetcher=fetch_match_weather,
    )


add_weather_impact_category = _weather.add_weather_impact_category
categorize_weather_impact = _weather.categorize_weather_impact

__all__ = [
    "add_weather_features", "add_weather_impact_category",
    "categorize_weather_impact", "fetch_match_weather",
]
