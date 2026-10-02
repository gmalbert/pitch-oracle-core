"""Reviewed aliases and unique oriented team pairs; never silently accept ambiguity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import pandas as pd

from pitch_oracle_core.domain.entities import normalized_name


@dataclass(frozen=True)
class FixtureMatch:
    fixture_id: str
    provider_match_id: str
    confidence: float
    reason: str


def normalize_name(value: str, aliases: dict[str, str]) -> str:
    lookup = {normalized_name(key): normalized_name(target) for key, target in aliases.items()}
    key = normalized_name(value)
    return lookup.get(key, key)


def _candidates(provider_row, canonical: pd.DataFrame, aliases: dict[str, str], max_hours: float):
    if max_hours <= 0:
        raise ValueError("Kickoff tolerance must be positive")
    required = {"fixture_id", "home_display_name", "away_display_name", "kickoff_utc"}
    if required.difference(canonical):
        raise ValueError(f"Canonical fixtures miss {sorted(required.difference(canonical))}")
    if canonical.fixture_id.isna().any() or canonical.fixture_id.duplicated().any():
        raise ValueError("Canonical fixture IDs must be present and unique")
    kickoff = pd.Timestamp(provider_row["kickoff_utc"])
    if kickoff.tzinfo is None or pd.isna(kickoff):
        raise ValueError("Provider kickoff must be a valid timezone-aware timestamp")
    home = normalize_name(str(provider_row["home_team"]), aliases)
    away = normalize_name(str(provider_row["away_team"]), aliases)
    frame = canonical.copy()
    frame["_home"] = frame.home_display_name.map(lambda value: normalize_name(str(value), aliases))
    frame["_away"] = frame.away_display_name.map(lambda value: normalize_name(str(value), aliases))
    if home == away:
        raise ValueError("Provider fixture team pair is invalid")
    times = pd.to_datetime(frame.kickoff_utc, utc=True, errors="raise")
    frame["_delta_h"] = (times - kickoff).abs().dt.total_seconds() / 3600
    # Filtering by edition/league when supplied prevents cross-competition collisions.
    for field in ("league_key", "edition_id"):
        if field in frame and pd.notna(provider_row.get(field)):
            frame = frame.loc[frame[field].astype(str) == str(provider_row[field])]
    nearby = frame.loc[frame._delta_h <= max_hours]
    oriented = nearby.loc[(nearby._home == home) & (nearby._away == away)]
    reversed_rows = nearby.loc[(nearby._home == away) & (nearby._away == home)]
    return oriented, reversed_rows


def map_provider_fixture(provider_row: pd.Series, canonical: pd.DataFrame, *, aliases: dict[str, str], max_hours: float = 6.0) -> FixtureMatch | None:
    oriented, _ = _candidates(provider_row, canonical, aliases, max_hours)
    if len(oriented) != 1:
        return None
    row = oriented.iloc[0]
    return FixtureMatch(str(row.fixture_id), str(provider_row["match_id"]), 1.0 - 0.2 * float(row._delta_h) / max_hours, "normalized oriented team pair + kickoff tolerance")


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
    for _, item in provider.iterrows():
        oriented, reversed_rows = _candidates(item, canonical, aliases, max_hours)
        status = "mapped" if len(oriented) == 1 else "ambiguous" if len(oriented) > 1 else "reversed" if len(reversed_rows) else "unmatched"
        if status == "mapped":
            row = oriented.iloc[0]
            match = map_provider_fixture(item, canonical, aliases=aliases, max_hours=max_hours)
            records.append({**asdict(match), "provider": "pitchapi", "home_team_id": row.get("home_team_id"), "away_team_id": row.get("away_team_id"), "kickoff_utc": row.kickoff_utc, "mapped_at": mapped_at.isoformat()})
        issues.append({"provider_match_id": str(item.match_id), "status": status, "home_team": item.home_team, "away_team": item.away_team, "kickoff_utc": item.kickoff_utc, "candidate_count": len(oriented), "kickoff_delta_hours": float(oriented.iloc[0]._delta_h) if len(oriented) == 1 else None})
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
