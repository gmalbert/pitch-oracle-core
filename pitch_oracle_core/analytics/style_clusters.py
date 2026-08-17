"""League-relative, explainable style fingerprints with fallback labels."""

from __future__ import annotations

import pandas as pd


def style_fingerprints(team_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"team_id", "goals_for", "goals_against"}
    missing = required.difference(team_metrics.columns)
    if missing:
        raise ValueError(f"Missing style inputs: {sorted(missing)}")
    frame = team_metrics.copy()
    tempo_source = "shots" if "shots" in frame.columns else "goals_for"
    possession_source = "possession" if "possession" in frame.columns else None
    frame["tempo_percentile"] = frame[tempo_source].rank(pct=True)
    frame["defense_percentile"] = frame.goals_against.rank(pct=True, ascending=False)
    if possession_source:
        frame["possession_percentile"] = frame[possession_source].rank(pct=True)
    else:
        frame["possession_percentile"] = 0.5

    def label(row) -> str:
        if row.tempo_percentile >= 0.75:
            return "high-event"
        if row.defense_percentile >= 0.75 and row.tempo_percentile < 0.5:
            return "low-block proxy"
        if row.possession_percentile >= 0.75:
            return "possession-control proxy"
        return "balanced"

    frame["style_label"] = frame.apply(label, axis=1)
    frame["input_coverage"] = "rich" if possession_source and tempo_source == "shots" else "aggregate fallback"
    frame["cluster_stability"] = 1.0
    frame["definition"] = (
        "Deterministic league-percentile style label over tempo, defensive, and "
        "available possession inputs; aggregate labels are explicitly proxies."
    )
    return frame
