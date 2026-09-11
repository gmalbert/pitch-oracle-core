"""MatchFlow event pipelines — lazy streaming for StatsBomb and Wyscout data.

Replaces eager ``pd.read_json`` over full archives with penaltyblog's
``Flow`` pipeline that scales to any dataset size.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from penaltyblog.matchflow import Flow, where_equals, where_in


def _records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return [payload] if isinstance(payload, dict) else []


def statsbomb_match_catalog(data_dir: str | Path) -> pd.DataFrame:
    """Read StatsBomb match metadata needed to scope event files safely."""
    rows: list[dict] = []
    for path in sorted((Path(data_dir) / "matches").glob("*.json")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for match in _records(path):
            competition = match.get("competition", {}) or {}
            season = match.get("season", {}) or {}
            home = match.get("home_team", {}) or {}
            away = match.get("away_team", {}) or {}
            rows.append({
                "match_id": match.get("match_id"),
                "competition_id": competition.get("competition_id"),
                "competition_name": competition.get("competition_name"),
                "season_id": season.get("season_id"),
                "season_name": season.get("season_name"),
                "match_date": match.get("match_date"),
                "kick_off": match.get("kick_off"),
                "home_team": home.get("home_team_name"),
                "away_team": away.get("away_team_name"),
                "metadata_file": str(path),
                "metadata_sha256": digest,
            })
    return pd.DataFrame(rows)


def statsbomb_source_manifest(data_dir: str | Path) -> pd.DataFrame:
    """Return immutable file metadata for event and match source snapshots."""
    root = Path(data_dir)
    rows = []
    for path in sorted(root.glob("events/*.json")) + sorted(root.glob("matches/*.json")):
        rows.append({
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        })
    return pd.DataFrame(rows, columns=["path", "sha256", "bytes"])


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
        catalog = statsbomb_match_catalog(data_dir)
        match_ids = catalog.loc[
            catalog["competition_id"] == competition_id, "match_id"
        ].dropna().tolist()
        if not match_ids:
            raise ValueError(f"no StatsBomb matches for competition_id={competition_id}")
        flow = flow.filter(where_in("match_id", match_ids))
    if season_id is not None:
        catalog = statsbomb_match_catalog(data_dir)
        match_ids = catalog.loc[
            catalog["season_id"] == season_id, "match_id"
        ].dropna().tolist()
        if not match_ids:
            raise ValueError(f"no StatsBomb matches for season_id={season_id}")
        flow = flow.filter(where_in("match_id", match_ids))
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
