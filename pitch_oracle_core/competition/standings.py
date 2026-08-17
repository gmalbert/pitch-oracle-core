"""Edition-versioned standings with deterministic tie handling."""

from __future__ import annotations

import pandas as pd

from pitch_oracle_core.domain.competitions import CompetitionRules, TieBreaker
from pitch_oracle_core.features.ledger import build_team_events


def _mini_table_points(matches: pd.DataFrame, teams: set[str], rules: CompetitionRules) -> dict[str, int]:
    subset = matches.loc[
        matches["home_team_id"].isin(teams) & matches["away_team_id"].isin(teams)
    ]
    if subset.empty:
        return {team: 0 for team in teams}
    events = build_team_events(subset)
    points = (
        (events["goal_diff"] > 0).astype(int) * rules.win_points
        + (events["goal_diff"] == 0).astype(int) * rules.draw_points
    )
    return points.groupby(events["team_id"]).sum().astype(int).to_dict()


def calculate_table(matches: pd.DataFrame, rules: CompetitionRules) -> pd.DataFrame:
    completed = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    if completed.empty:
        return pd.DataFrame(columns=[
            "position", "team_id", "played", "wins", "draws", "losses",
            "goals_for", "goals_against", "goal_difference", "points",
        ])
    events = build_team_events(completed)
    events["win"] = (events["goal_diff"] > 0).astype(int)
    events["draw"] = (events["goal_diff"] == 0).astype(int)
    events["loss"] = (events["goal_diff"] < 0).astype(int)
    events["points"] = events["win"] * rules.win_points + events["draw"] * rules.draw_points
    table = events.groupby("team_id", as_index=False).agg(
        played=("fixture_id", "count"),
        wins=("win", "sum"),
        draws=("draw", "sum"),
        losses=("loss", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        points=("points", "sum"),
    )
    table["goal_difference"] = table["goals_for"] - table["goals_against"]
    table["points"] += table["team_id"].map(rules.points_adjustments).fillna(0)
    if TieBreaker.HEAD_TO_HEAD_POINTS in rules.tie_breakers:
        table["head_to_head_points"] = 0
        for _, tied in table.groupby("points"):
            if len(tied) > 1:
                mini = _mini_table_points(completed, set(tied["team_id"]), rules)
                table.loc[tied.index, "head_to_head_points"] = tied["team_id"].map(mini)
    sort_columns = [item.value for item in rules.tie_breakers]
    unsupported = set(sort_columns).difference(table.columns)
    if unsupported:
        raise NotImplementedError(
            f"Tie-breakers require a specialized resolver: {sorted(unsupported)}"
        )
    table = table.sort_values(
        [*sort_columns, "team_id"],
        ascending=[False] * len(sort_columns) + [True],
        kind="stable",
    ).reset_index(drop=True)
    table.insert(0, "position", table.index + 1)
    return table
