"""Point-in-time fixture difficulty calendars."""

from __future__ import annotations

import pandas as pd


def fixture_difficulty(
    team_fixtures: pd.DataFrame, rating_history: pd.DataFrame
) -> pd.DataFrame:
    required = {"fixture_id", "team_id", "opponent_id", "venue_role", "kickoff_utc"}
    missing = required.difference(team_fixtures.columns)
    if missing:
        raise ValueError(f"Missing fixture difficulty columns: {sorted(missing)}")
    rating_column = (
        "pre_match_rating" if "pre_match_rating" in rating_history else "rating"
    )
    if rating_column not in rating_history:
        raise ValueError("Rating history requires a pre-match rating")
    rating_columns = ["fixture_id", "team_id", rating_column]
    if "observed_at" in rating_history:
        rating_columns.append("observed_at")
    ratings = rating_history[rating_columns].rename(
        columns={"team_id": "opponent_id", rating_column: "opponent_rating"}
    )
    result = team_fixtures.merge(
        ratings, on=["fixture_id", "opponent_id"], how="left", validate="many_to_one"
    )
    if "observed_at" in result:
        observed = pd.to_datetime(result.observed_at, utc=True, errors="coerce")
        kickoff = pd.to_datetime(result.kickoff_utc, utc=True, errors="coerce")
        if (observed.notna() & (observed >= kickoff)).any():
            raise ValueError("Fixture difficulty contains a post-kickoff rating")
    result["opponent_rating"] = result.opponent_rating.fillna(1500.0)
    venue_bonus = result.venue_role.map({"home": -55.0, "away": 55.0}).fillna(0.0)
    result["difficulty_rating"] = result.opponent_rating + venue_bonus
    result["difficulty_percentile"] = result.groupby("team_id")["difficulty_rating"].rank(pct=True)
    result["rating_observed_at"] = result.get("observed_at", pd.NaT)
    return result
