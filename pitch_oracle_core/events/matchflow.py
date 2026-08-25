"""MatchFlow event pipelines — lazy streaming for StatsBomb and Wyscout data.

Replaces eager ``pd.read_json`` over full archives with penaltyblog's
``Flow`` pipeline that scales to any dataset size.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from penaltyblog.matchflow import Flow, where_equals, where_gt


def statsbomb_events(
    data_dir: str | Path,
    competition_id: int | None = None,
    season_id: int | None = None,
) -> Flow:
    """Stream StatsBomb open-data events from local JSON files.

    Returns a ``Flow`` that can be further filtered, selected, or collected.
    """
    pattern = str(Path(data_dir) / "events" / "*.json")
    flow = Flow.from_glob(pattern)
    if competition_id is not None:
        flow = flow.filter(where_equals("competition_id", competition_id))
    if season_id is not None:
        flow = flow.filter(where_equals("season_id", season_id))
    return flow


def statsbomb_shots(data_dir: str | Path, **kwargs) -> pd.DataFrame:
    """Extract all shots from StatsBomb open data as a DataFrame."""
    return (
        statsbomb_events(data_dir, **kwargs)
        .filter(where_equals("type.name", "Shot"))
        .select("match_id", "team", "player", "type", "minute", "location", "shot")
        .to_pandas()
    )


def statsbomb_passes(data_dir: str | Path, **kwargs) -> pd.DataFrame:
    """Extract all passes from StatsBomb open data as a DataFrame."""
    return (
        statsbomb_events(data_dir, **kwargs)
        .filter(where_equals("type.name", "Pass"))
        .select("match_id", "team", "player", "type", "minute", "location", "pass")
        .to_pandas()
    )


def wyscout_events(data_dir: str | Path) -> Flow:
    """Stream Wyscout events from local JSON/JSONL files.

    Returns a ``Flow`` that can be further filtered, selected, or collected.
    """
    pattern = str(Path(data_dir) / "*.json")
    return Flow.from_glob(pattern)


def wyscout_shots(data_dir: str | Path) -> pd.DataFrame:
    """Extract all shots from Wyscout data as a DataFrame."""
    return (
        wyscout_events(data_dir)
        .filter(where_equals("eventName", "Shot"))
        .to_pandas()
    )


def match_summary(events: pd.DataFrame, match_id: str) -> dict[str, int]:
    """Compute per-match event summary from a collected events DataFrame."""
    match = events[events["match_id"] == match_id] if "match_id" in events.columns else events
    return {
        "total_events": len(match),
        "shots": int((match.get("type.name", match.get("eventName", "")) == "Shot").sum())
              if "type.name" in match.columns or "eventName" in match.columns else 0,
        "passes": int((match.get("type.name", match.get("eventName", "")) == "Pass").sum())
              if "type.name" in match.columns or "eventName" in match.columns else 0,
    }
