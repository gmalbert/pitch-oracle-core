"""Congestion, travel, and recovery load."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
import numpy as np
import pandas as pd


def haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    values = (lat_a, lon_a, lat_b, lon_b)
    if not all(np.isfinite(values)):
        raise ValueError("coordinates must be finite")
    if not (-90 <= lat_a <= 90 and -90 <= lat_b <= 90):
        raise ValueError("latitude outside [-90, 90]")
    if not (-180 <= lon_a <= 180 and -180 <= lon_b <= 180):
        raise ValueError("longitude outside [-180, 180]")
    radius_km = 6371.0088
    d_lat, d_lon = radians(lat_b - lat_a), radians(lon_b - lon_a)
    value = (
        sin(d_lat / 2) ** 2
        + cos(radians(lat_a)) * cos(radians(lat_b)) * sin(d_lon / 2) ** 2
    )
    return 2 * radius_km * asin(sqrt(value))


def recovery_features(team_fixtures: pd.DataFrame) -> pd.DataFrame:
    required = {"team_id", "kickoff_utc"}
    missing = required.difference(team_fixtures.columns)
    if missing:
        raise ValueError(f"Missing schedule columns: {sorted(missing)}")
    frame = team_fixtures.copy()
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True)
    frame = frame.sort_values(["team_id", "kickoff_utc"])
    group = frame.groupby("team_id", sort=False)
    frame["days_since_previous"] = group.kickoff_utc.diff().dt.total_seconds() / 86_400

    def count_previous_14d(series: pd.Series) -> pd.Series:
        values = series.astype("int64").to_numpy()
        horizon_ns = 14 * 86_400 * 1_000_000_000
        counts = [
            index - np.searchsorted(values, value - horizon_ns, side="left")
            for index, value in enumerate(values)
        ]
        return pd.Series(counts, index=series.index, dtype=int)

    frame["matches_previous_14d"] = group.kickoff_utc.transform(count_previous_14d)
    frame["short_rest"] = frame.days_since_previous < 4
    coordinate_columns = {
        "previous_venue_lat", "previous_venue_lon", "venue_lat", "venue_lon"
    }
    if coordinate_columns.issubset(frame.columns):
        trusted = {"verified", "venue_verified", "high"}
        frame["travel_km"] = frame.apply(
            lambda row: haversine_km(
                row.previous_venue_lat,
                row.previous_venue_lon,
                row.venue_lat,
                row.venue_lon,
            )
            if pd.notna(row.previous_venue_lat)
            and pd.notna(row.previous_venue_lon)
            and pd.notna(row.venue_lat)
            and pd.notna(row.venue_lon)
            and (
                "venue_confidence" not in frame
                or str(row.venue_confidence).casefold() in trusted
            )
            and (
                "previous_venue_confidence" not in frame
                or str(row.previous_venue_confidence).casefold() in trusted
            )
            else float("nan"),
            axis=1,
        )
    else:
        frame["travel_km"] = np.nan
    frame["recovery_load"] = (
        frame["short_rest"].astype(float)
        + frame["matches_previous_14d"].clip(lower=0).div(5)
        + frame["travel_km"].fillna(0).div(2_000).clip(upper=1)
    )
    frame["travel_confidence"] = np.where(
        frame.travel_km.notna(), "venue_verified", "omitted_low_or_missing_confidence"
    )
    return frame
