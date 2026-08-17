"""Team command-center snapshots from one chronological team ledger."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _shrunk_mean(values: pd.Series, league_mean: float, prior_matches: float = 5.0) -> float:
    observed = values.dropna().astype(float)
    return float((observed.sum() + league_mean * prior_matches) / (len(observed) + prior_matches))


def build_team_snapshots(
    events: pd.DataFrame,
    *,
    as_of: object | None = None,
    elo_ratings: dict[str, float] | None = None,
) -> pd.DataFrame:
    required = {
        "team_id", "fixture_id", "kickoff_utc", "venue_role", "goals_for",
        "goals_against", "points", "opponent_id",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing team-event columns: {sorted(missing)}")
    frame = events.copy()
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True)
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        frame = frame.loc[frame["kickoff_utc"] < cutoff]
    frame = frame.dropna(subset=["goals_for", "goals_against", "points"])
    league_for = float(frame["goals_for"].mean()) if not frame.empty else 0.0
    rows = []
    for team_id, group in frame.groupby("team_id", sort=True):
        group = group.sort_values("kickoff_utc")
        recent5, recent10 = group.tail(5), group.tail(10)
        home, away = group.loc[group.venue_role == "home"], group.loc[group.venue_role == "away"]
        row = {
            "team_id": team_id,
            "matches": len(group),
            "points": float(group.points.sum()),
            "points_l5": float(recent5.points.sum()),
            "points_per_match_l10": float(recent10.points.mean()),
            "attack_l10": float(recent10.goals_for.mean()),
            "defense_l10": float(recent10.goals_against.mean()),
            "goal_difference_l10": float((recent10.goals_for - recent10.goals_against).mean()),
            "clean_sheet_rate_l10": float((recent10.goals_against == 0).mean()),
            "home_goals_for_shrunk": _shrunk_mean(home.goals_for, league_for),
            "away_goals_for_shrunk": _shrunk_mean(away.goals_for, league_for),
            "home_sample": len(home),
            "away_sample": len(away),
            "elo_rating": (elo_ratings or {}).get(team_id, 1500.0),
            "latest_kickoff": group.kickoff_utc.max(),
        }
        for source, target in (
            ("xg_for", "xg_for_ewm10"),
            ("xg_against", "xg_against_ewm10"),
            ("shots_for", "shots_for_ewm10"),
            ("shot_quality", "shot_quality_ewm10"),
            ("finishing_vs_expectation", "finishing_vs_expectation_ewm10"),
            ("opponent_adjusted_points", "opponent_adjusted_points_l10"),
        ):
            if source in recent10:
                values = pd.to_numeric(recent10[source], errors="coerce")
                weights = np.exp(np.linspace(-2.0, 0.0, len(values)))
                valid = values.notna().to_numpy()
                row[target] = (
                    float(np.average(values.to_numpy()[valid], weights=weights[valid]))
                    if valid.any() else np.nan
                )
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["power_rank"] = result["elo_rating"].rank(
            ascending=False, method="min"
        ).astype(int)
    return result


def opponent_adjusted_performance(
    events: pd.DataFrame, rating_history: pd.DataFrame
) -> pd.DataFrame:
    ratings = rating_history[["fixture_id", "team_id", "pre_match_rating"]].rename(
        columns={"team_id": "opponent_id", "pre_match_rating": "opponent_rating"}
    )
    joined = events.merge(
        ratings, on=["fixture_id", "opponent_id"], how="left", validate="many_to_one"
    )
    expected_points = 3 / (1 + 10 ** (-(joined.opponent_rating.fillna(1500) - 1500) / 400))
    joined["opponent_adjusted_points"] = joined.points - expected_points
    return joined
