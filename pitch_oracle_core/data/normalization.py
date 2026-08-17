"""Normalize provider fixtures into the canonical fixture contract."""

from __future__ import annotations

from datetime import date
import hashlib
import pandas as pd

from pitch_oracle_core.domain.competitions import CompetitionEdition
from pitch_oracle_core.domain.entities import EntityResolver, Resolution


def stable_fixture_id(
    edition_id: str, provider: str, provider_event_id: str | int
) -> str:
    digest = hashlib.sha256(
        f"{edition_id}|{provider}|{provider_event_id}".encode("utf-8")
    ).hexdigest()[:20]
    return f"fx:{digest}"


def normalize_fixtures(
    source: pd.DataFrame,
    *,
    edition: CompetitionEdition,
    resolver: EntityResolver,
    provider: str,
    columns: dict[str, str] | None = None,
    observed_at: object | None = None,
) -> tuple[pd.DataFrame, list[Resolution]]:
    mapping = {
        "provider_event_id": "provider_event_id",
        "kickoff": "kickoff_utc",
        "home": "home_team",
        "away": "away_team",
        "status": "status",
        "venue_id": "venue_id",
    }
    mapping.update(columns or {})
    needed = {mapping[key] for key in ("provider_event_id", "kickoff", "home", "away")}
    missing = needed.difference(source.columns)
    if missing:
        raise ValueError(f"Fixture source is missing columns: {sorted(missing)}")
    frame = source.copy()
    frame["kickoff_utc"] = pd.to_datetime(frame[mapping["kickoff"]], utc=True, errors="coerce")
    if frame["kickoff_utc"].isna().any():
        raise ValueError("Every fixture requires a timezone-aware kickoff")
    resolutions: list[Resolution] = []
    rows: list[dict[str, object]] = []
    default_observed = pd.Timestamp(observed_at or pd.Timestamp.now(tz="UTC"))
    if default_observed.tzinfo is None:
        default_observed = default_observed.tz_localize("UTC")
    for item in frame.itertuples(index=False):
        raw = item._asdict()
        kickoff = pd.Timestamp(raw[mapping["kickoff"]])
        if kickoff.tzinfo is None:
            kickoff = kickoff.tz_localize("UTC")
        when: date = kickoff.date()
        home = resolver.resolve(provider, str(raw[mapping["home"]]), when)
        away = resolver.resolve(provider, str(raw[mapping["away"]]), when)
        resolutions.extend((home, away))
        event_id = raw[mapping["provider_event_id"]]
        rows.append(
            {
                "fixture_id": stable_fixture_id(edition.edition_id, provider, event_id),
                "edition_id": edition.edition_id,
                "rules_version": edition.rules_version,
                "kickoff_utc": kickoff.tz_convert("UTC"),
                "home_team_id": home.team_id,
                "away_team_id": away.team_id,
                "home_display_name": raw[mapping["home"]],
                "away_display_name": raw[mapping["away"]],
                "home_resolution_status": home.status.value,
                "away_resolution_status": away.status.value,
                "status": raw.get(mapping["status"], "scheduled"),
                "venue_id": raw.get(mapping["venue_id"]),
                "provider_event_id": str(event_id),
                "source": provider,
                "observed_at": default_observed,
            }
        )
    return pd.DataFrame(rows), resolutions
