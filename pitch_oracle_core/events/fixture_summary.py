"""Per-fixture summary artifact via MatchFlow — event counts and totals.

Aggregates event-level data into per-fixture summaries that feed the
Match Centre header and other UI components.
"""

from __future__ import annotations

import pandas as pd


def fixture_event_summary(
    events: pd.DataFrame,
    match_id_col: str = "match_id",
    type_col: str = "type.name",
) -> pd.DataFrame:
    """Aggregate events into per-fixture summaries."""
    summaries = []
    for match_id, group in events.groupby(match_id_col):
        summary = {"match_id": match_id, "total_events": len(group)}
        if type_col in group.columns:
            type_counts = group[type_col].value_counts()
            for event_type in ("Shot", "Pass", "Duel", "Foul", "Ball Recovery",
                               "Clearance", "Interception", "Goal Keeper"):
                summary[f"n_{event_type.lower().replace(' ', '_')}"] = int(type_counts.get(event_type, 0))
        if "minute" in group.columns:
            summary["first_event_minute"] = int(group["minute"].min())
            summary["last_event_minute"] = int(group["minute"].max())
        summaries.append(summary)
    return pd.DataFrame(summaries)


def shot_summary(
    shots: pd.DataFrame,
    match_id_col: str = "match_id",
    xg_col: str = "shot.statsbomb_xg",
    team_col: str = "team.name",
) -> pd.DataFrame:
    """Aggregate shot-level data into per-fixture xG summaries."""
    if xg_col not in shots.columns:
        return pd.DataFrame()
    rows = []
    for match_id, group in shots.groupby(match_id_col):
        row = {"match_id": match_id, "total_shots": len(group)}
        if team_col in group.columns:
            teams = group[team_col].unique()
            if len(teams) >= 2:
                home_shots = group[group[team_col] == teams[0]]
                away_shots = group[group[team_col] == teams[1]]
                row["home_xg"] = float(home_shots[xg_col].sum())
                row["away_xg"] = float(away_shots[xg_col].sum())
                row["home_shots"] = len(home_shots)
                row["away_shots"] = len(away_shots)
        rows.append(row)
    return pd.DataFrame(rows)
