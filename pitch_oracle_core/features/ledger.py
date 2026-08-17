"""Perspective-normalized team-event ledger and pre-match state."""

from __future__ import annotations

import numpy as np
import pandas as pd


MATCH_COLUMNS = {
    "fixture_id",
    "edition_id",
    "kickoff_utc",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
}


def build_team_events(matches: pd.DataFrame) -> pd.DataFrame:
    missing = MATCH_COLUMNS.difference(matches.columns)
    if missing:
        raise ValueError(f"Missing match columns: {sorted(missing)}")
    frame = matches.copy()
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True, errors="coerce")
    if frame["kickoff_utc"].isna().any():
        raise ValueError("Every match requires a valid kickoff_utc")
    if frame["fixture_id"].isna().any() or frame["fixture_id"].duplicated().any():
        raise ValueError("fixture_id must be present and unique")
    if (frame["home_team_id"] == frame["away_team_id"]).any():
        raise ValueError("A team cannot play itself")

    home = pd.DataFrame(
        {
            "fixture_id": frame["fixture_id"],
            "edition_id": frame["edition_id"],
            "kickoff_utc": frame["kickoff_utc"],
            "team_id": frame["home_team_id"],
            "opponent_id": frame["away_team_id"],
            "venue_role": "home",
            "goals_for": pd.to_numeric(frame["home_goals"], errors="coerce"),
            "goals_against": pd.to_numeric(frame["away_goals"], errors="coerce"),
        }
    )
    away = pd.DataFrame(
        {
            "fixture_id": frame["fixture_id"],
            "edition_id": frame["edition_id"],
            "kickoff_utc": frame["kickoff_utc"],
            "team_id": frame["away_team_id"],
            "opponent_id": frame["home_team_id"],
            "venue_role": "away",
            "goals_for": pd.to_numeric(frame["away_goals"], errors="coerce"),
            "goals_against": pd.to_numeric(frame["home_goals"], errors="coerce"),
        }
    )
    optional_pairs = {
        "xg": (("home_xg", "xg_home"), ("away_xg", "xg_away")),
        "shots": (("home_shots", "shots_home"), ("away_shots", "shots_away")),
        "shots_on_target": (
            ("home_shots_on_target", "shots_on_target_home"),
            ("away_shots_on_target", "shots_on_target_away"),
        ),
    }
    for metric, (home_candidates, away_candidates) in optional_pairs.items():
        home_source = next((name for name in home_candidates if name in frame), None)
        away_source = next((name for name in away_candidates if name in frame), None)
        if home_source is not None and away_source is not None:
            home[f"{metric}_for"] = pd.to_numeric(frame[home_source], errors="coerce")
            home[f"{metric}_against"] = pd.to_numeric(frame[away_source], errors="coerce")
            away[f"{metric}_for"] = pd.to_numeric(frame[away_source], errors="coerce")
            away[f"{metric}_against"] = pd.to_numeric(frame[home_source], errors="coerce")
    events = pd.concat([home, away], ignore_index=True)
    incomplete = events["goals_for"].isna() ^ events["goals_against"].isna()
    if incomplete.any():
        raise ValueError("A result must provide both home and away goals")
    if (events[["goals_for", "goals_against"]].dropna() < 0).any().any():
        raise ValueError("Goals cannot be negative")
    events["goal_diff"] = events["goals_for"] - events["goals_against"]
    events["points"] = np.select(
        [events["goal_diff"] > 0, events["goal_diff"] == 0, events["goal_diff"] < 0],
        [3.0, 1.0, 0.0],
        default=np.nan,
    )
    events["result"] = np.select(
        [events["goal_diff"] > 0, events["goal_diff"] == 0, events["goal_diff"] < 0],
        ["W", "D", "L"],
        default=None,
    )
    if "xg_for" in events:
        events["finishing_vs_expectation"] = events["goals_for"] - events["xg_for"]
    if "shots_for" in events:
        denominator = events["shots_for"].where(events["shots_for"] > 0)
        numerator = events.get("xg_for", events["goals_for"])
        events["shot_quality"] = numerator / denominator
    return events.sort_values(
        ["team_id", "kickoff_utc", "fixture_id"], kind="stable"
    ).reset_index(drop=True)


def _prior_rolling(
    frame: pd.DataFrame, value: str, window: int, aggregation: str
) -> pd.Series:
    grouped = frame.groupby("team_id", sort=False)[value]
    if aggregation == "sum":
        return grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=1).sum()
        )
    if aggregation == "mean":
        return grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=1).mean()
        )
    raise ValueError(f"Unsupported aggregation {aggregation!r}")


def _prior_ewm(frame: pd.DataFrame, value: str, span: int = 10) -> pd.Series:
    """Exponentially weighted pre-match value, never including the current event."""
    return frame.groupby("team_id", sort=False)[value].transform(
        lambda series: series.shift(1).ewm(span=span, adjust=False, min_periods=1).mean()
    )


def add_prior_team_state(events: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fixture_id", "kickoff_utc", "team_id", "goals_for", "goals_against",
        "goal_diff", "points",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    state = events.sort_values(
        ["team_id", "kickoff_utc", "fixture_id"], kind="stable"
    ).copy()
    state["history_n"] = state.groupby("team_id", sort=False)["goals_for"].transform(
        lambda series: series.shift(1).notna().cumsum()
    )
    state["rest_days"] = (
        state.groupby("team_id", sort=False)["kickoff_utc"]
        .diff()
        .dt.total_seconds()
        .div(86_400)
    )
    state["points_l5"] = _prior_rolling(state, "points", 5, "sum")
    state["goals_for_l5"] = _prior_rolling(state, "goals_for", 5, "mean")
    state["goals_against_l5"] = _prior_rolling(state, "goals_against", 5, "mean")
    state["goal_diff_l10"] = _prior_rolling(state, "goal_diff", 10, "mean")
    state["clean_sheet"] = np.where(
        state["goals_against"].notna(),
        (state["goals_against"] == 0).astype(float),
        np.nan,
    )
    state["clean_sheet_l10"] = _prior_rolling(state, "clean_sheet", 10, "mean")
    for source, target in (
        ("xg_for", "xg_for_ewm10"),
        ("xg_against", "xg_against_ewm10"),
        ("shots_for", "shots_for_ewm10"),
        ("shots_against", "shots_against_ewm10"),
        ("shot_quality", "shot_quality_ewm10"),
        ("finishing_vs_expectation", "finishing_vs_expectation_ewm10"),
    ):
        if source in state:
            state[target] = _prior_ewm(state, source)
    return state


def match_feature_snapshots(events: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "fixture_id",
        "history_n",
        "rest_days",
        "points_l5",
        "goals_for_l5",
        "goals_against_l5",
        "goal_diff_l10",
        "clean_sheet_l10",
    ]
    missing = set(feature_columns).difference(events.columns)
    if missing:
        raise ValueError(f"Missing state columns: {sorted(missing)}")
    optional = [
        name for name in (
            "xg_for_ewm10", "xg_against_ewm10", "shots_for_ewm10",
            "shots_against_ewm10", "shot_quality_ewm10",
            "finishing_vs_expectation_ewm10",
        ) if name in events
    ]
    feature_columns.extend(optional)
    home = events.loc[events["venue_role"] == "home", feature_columns].add_prefix("home_")
    away = events.loc[events["venue_role"] == "away", feature_columns].add_prefix("away_")
    result = home.merge(
        away,
        left_on="home_fixture_id",
        right_on="away_fixture_id",
        validate="one_to_one",
    )
    return result.rename(columns={"home_fixture_id": "fixture_id"}).drop(
        columns="away_fixture_id"
    )
