"""Deterministic matchday storylines with supporting evidence."""

from __future__ import annotations

import pandas as pd


def matchday_storylines(events: pd.DataFrame) -> pd.DataFrame:
    definitions = (
        ("rating_move", "Biggest power-rating move", True),
        ("upset_index", "Largest model upset", True),
        ("form_swing", "Sharpest form swing", True),
        ("stakes_index", "Highest-stakes fixture", True),
        ("surprise_score", "Biggest model surprise", True),
    )
    rows = []
    for metric, label, descending in definitions:
        if metric not in events.columns or events[metric].dropna().empty:
            continue
        ordered = events.sort_values(metric, ascending=not descending)
        item = ordered.iloc[0]
        rows.append({
            "storyline": label,
            "fixture_id": item.get("fixture_id"),
            "team_id": item.get("team_id"),
            "metric": metric,
            "value": float(item[metric]),
            "link_path": (
                "/prediction-history?fixture="
                if str(item.get("status", "scheduled")).casefold() in {"completed", "final"}
                else "/match-center?fixture="
            ),
        })
    return pd.DataFrame(rows)
