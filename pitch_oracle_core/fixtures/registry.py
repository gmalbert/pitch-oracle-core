"""Persist canonical identity across source transitions and kickoff corrections."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from pitch_oracle_core.pitchapi.cache import atomic_json


def assign_fixture_ids(frame: pd.DataFrame, path: str | Path, *, league_key: str, persist: bool = True) -> pd.DataFrame:
    path = Path(path)
    registry = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"schema_version": 1, "league_key": league_key, "fixtures": {}}
    if registry.get("schema_version") != 1 or registry.get("league_key") != league_key:
        raise ValueError("Canonical registry belongs to a different league or schema")
    entries = registry["fixtures"]
    aliases = {}
    oriented = {}
    for fixture_id, entry in entries.items():
        oriented.setdefault(tuple(entry[field] for field in ("edition_id", "home_team_id", "away_team_id")), []).append(fixture_id)
        for alias in entry["aliases"]:
            if alias in aliases and aliases[alias] != fixture_id:
                raise ValueError("Canonical registry has ambiguous source aliases")
            aliases[alias] = fixture_id
    result = frame.copy()
    for index, row in result.iterrows():
        provided = str(row.fixture_id)
        pair = {field: str(row[field]) for field in ("edition_id", "home_team_id", "away_team_id")}
        kickoff = pd.Timestamp(row.kickoff_utc)
        if kickoff.tzinfo is None:
            raise ValueError("Registry kickoff must be timezone-aware")
        candidate = aliases.get(provided)
        if candidate is not None and any(entries[candidate][field] != value for field, value in pair.items()):
            raise ValueError("Source fixture identity changed competition or oriented teams")
        if candidate is None:
            possible = []
            for fixture_id in oriented.get(tuple(pair.values()), []):
                entry = entries[fixture_id]
                for known in entry["kickoffs"]:
                    source_time = pd.Timestamp(known["kickoff_utc"])
                    date_only = row.get("kickoff_precision") == "date_only" or known["precision"] == "date_only"
                    zone = str(row.get("source_clock_timezone", "UTC")) if row.get("kickoff_precision") == "date_only" else known.get("source_clock_timezone", "UTC")
                    same_day = source_time.tz_convert(zone).date() == kickoff.tz_convert(zone).date()
                    if (date_only and same_day) or (not date_only and abs(source_time - kickoff) <= pd.Timedelta(hours=6)):
                        possible.append(fixture_id)
                        break
            if len(possible) > 1:
                raise ValueError("Ambiguous canonical source transition")
            candidate = possible[0] if possible else provided
        if candidate not in entries:
            entries[candidate] = {**pair, "aliases": [], "kickoffs": []}
            oriented.setdefault(tuple(pair.values()), []).append(candidate)
        entry = entries[candidate]
        if provided not in entry["aliases"]:
            entry["aliases"].append(provided)
        aliases[provided] = candidate
        known = {"kickoff_utc": kickoff.isoformat(), "precision": str(row.get("kickoff_precision", "exact")), "source_clock_timezone": str(row.get("source_clock_timezone", "UTC"))}
        if known not in entry["kickoffs"]:
            entry["kickoffs"].append(known)
        result.at[index, "fixture_id"] = candidate
    if result.fixture_id.duplicated().any():
        raise ValueError("Canonical source contains duplicate events after registry reconciliation")
    if persist:
        registry["checked_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(path, registry)
    return result
