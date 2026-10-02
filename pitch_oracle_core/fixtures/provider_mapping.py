"""Reviewed aliases and unique oriented team pairs; never silently accept ambiguity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import pandas as pd

from pitch_oracle_core.domain.entities import normalized_name


@dataclass(frozen=True)
class FixtureMatch:
    fixture_id: str
    provider_match_id: str
    confidence: float
    reason: str


def normalize_name(value: str, aliases: dict[str, str]) -> str:
    lookup = _alias_lookup(tuple(sorted(aliases.items())))
    key = normalized_name(value)
    return lookup.get(key, key)


@lru_cache(maxsize=32)
def _alias_lookup(aliases: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return {normalized_name(key): normalized_name(target) for key, target in aliases}


def _prepare_canonical(canonical: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    required = {"fixture_id", "home_display_name", "away_display_name", "kickoff_utc"}
    if required.difference(canonical):
        raise ValueError(f"Canonical fixtures miss {sorted(required.difference(canonical))}")
    if canonical.fixture_id.isna().any() or canonical.fixture_id.duplicated().any():
        raise ValueError("Canonical fixture IDs must be present and unique")
    frame = canonical.copy()
    lookup = _alias_lookup(tuple(sorted(aliases.items())))
    def name(value):
        key = normalized_name(str(value))
        return lookup.get(key, key)
    frame["_home"] = frame.home_display_name.map(name)
    frame["_away"] = frame.away_display_name.map(name)
    frame["_kickoff_time"] = pd.to_datetime(frame.kickoff_utc, utc=True, errors="raise")
    return frame


def _candidates(provider_row, canonical: pd.DataFrame, aliases: dict[str, str], max_hours: float):
    if max_hours <= 0:
        raise ValueError("Kickoff tolerance must be positive")
    kickoff = pd.Timestamp(provider_row["kickoff_utc"])
    if kickoff.tzinfo is None or pd.isna(kickoff):
        raise ValueError("Provider kickoff must be a valid timezone-aware timestamp")
    home = normalize_name(str(provider_row["home_team"]), aliases)
    away = normalize_name(str(provider_row["away_team"]), aliases)
    prepared = canonical if {"_home", "_away", "_kickoff_time"}.issubset(canonical) else _prepare_canonical(canonical, aliases)
    pairs = ((prepared._home == home) & (prepared._away == away)) | ((prepared._home == away) & (prepared._away == home))
    frame = prepared.loc[pairs].copy()
    if home == away:
        raise ValueError("Provider fixture team pair is invalid")
    frame["_delta_h"] = (frame._kickoff_time - kickoff).abs().dt.total_seconds() / 3600
    # Filtering by edition/league when supplied prevents cross-competition collisions.
    for field in ("league_key", "edition_id"):
        if field in frame and pd.notna(provider_row.get(field)):
            frame = frame.loc[frame[field].astype(str) == str(provider_row[field])]
    date_only = frame.get("kickoff_precision", pd.Series("exact", index=frame.index)).eq("date_only")
    date_match = pd.Series(False, index=frame.index)
    for index, row in frame.loc[date_only].iterrows():
        zone = row.get("source_clock_timezone", "UTC")
        date_match.loc[index] = kickoff.tz_convert(zone).date() == row._kickoff_time.tz_convert(zone).date()
    nearby = frame.loc[(~date_only & (frame._delta_h <= max_hours)) | (date_only & date_match)]
    oriented = nearby.loc[(nearby._home == home) & (nearby._away == away)]
    reversed_rows = nearby.loc[(nearby._home == away) & (nearby._away == home)]
    return oriented, reversed_rows


def map_provider_fixture(provider_row: pd.Series, canonical: pd.DataFrame, *, aliases: dict[str, str], max_hours: float = 6.0) -> FixtureMatch | None:
    oriented, _ = _candidates(provider_row, canonical, aliases, max_hours)
    if len(oriented) != 1:
        return None
    row = oriented.iloc[0]
    date_only = row.get("kickoff_precision") == "date_only"
    return FixtureMatch(str(row.fixture_id), str(provider_row["match_id"]), .8 if date_only else 1.0 - 0.2 * float(row._delta_h) / max_hours, "normalized oriented team pair + date-only source" if date_only else "normalized oriented team pair + kickoff tolerance")


def reconcile_fixtures(
    provider: pd.DataFrame, canonical: pd.DataFrame, *, aliases: dict[str, str],
    max_hours: float = 6.0, mapped_at: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped_at = mapped_at or datetime.now(timezone.utc)
    if mapped_at.tzinfo is None:
        raise ValueError("Mapping timestamp must be timezone-aware")
    if "match_id" not in provider:
        raise ValueError("Provider fixtures require match_id")
    if provider.match_id.isna().any() or provider.match_id.duplicated().any():
        raise ValueError("Provider match IDs must be present and unique")
    records, issues = [], []
    prepared = _prepare_canonical(canonical, aliases)
    for _, item in provider.iterrows():
        oriented, reversed_rows = _candidates(item, prepared, aliases, max_hours)
        status = "mapped" if len(oriented) == 1 else "ambiguous" if len(oriented) > 1 else "reversed" if len(reversed_rows) else "unmatched"
        if status == "mapped":
            row = oriented.iloc[0]
            date_only = row.get("kickoff_precision") == "date_only"
            match = FixtureMatch(str(row.fixture_id), str(item.match_id), .8 if date_only else 1.0 - 0.2 * float(row._delta_h) / max_hours, "normalized oriented team pair + date-only source" if date_only else "normalized oriented team pair + kickoff tolerance")
            records.append({**asdict(match), "provider": "pitchapi", "home_team_id": row.get("home_team_id"), "away_team_id": row.get("away_team_id"), "kickoff_utc": row.kickoff_utc, "mapped_at": mapped_at.isoformat()})
        exact = len(oriented) == 1 and oriented.iloc[0].get("kickoff_precision") != "date_only"
        issues.append({"provider_match_id": str(item.match_id), "status": status, "home_team": item.home_team, "away_team": item.away_team, "kickoff_utc": item.kickoff_utc, "candidate_count": len(oriented), "kickoff_delta_hours": float(oriented.iloc[0]._delta_h) if exact else None, "kickoff_comparison": "exact" if exact else "date_only" if len(oriented) == 1 else "unresolved"})
    columns = ["fixture_id", "provider_match_id", "confidence", "reason", "provider", "home_team_id", "away_team_id", "kickoff_utc", "mapped_at"]
    result = pd.DataFrame(records, columns=columns)
    # A canonical event cannot be silently assigned two provider events.
    duplicated = result.fixture_id.duplicated(keep=False)
    if duplicated.any():
        rejected = set(result.loc[duplicated, "provider_match_id"])
        result = result.loc[~duplicated].copy()
        for issue in issues:
            if issue["provider_match_id"] in rejected:
                issue["status"] = "ambiguous"
    return result.reset_index(drop=True), pd.DataFrame(issues)
