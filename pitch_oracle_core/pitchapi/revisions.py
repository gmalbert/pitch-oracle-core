"""Complete response revisions, including authoritative removal of prior rows."""

from __future__ import annotations

import pandas as pd


def with_removals(incoming: pd.DataFrame, previous: pd.DataFrame, *, fixture_id: str, identity: str, observed_at) -> pd.DataFrame:
    """Keep removals as null observations so past cutoffs remain reproducible."""
    incoming = incoming.copy().assign(is_deleted=False)
    if previous.empty:
        return incoming
    history = previous.loc[previous.fixture_id.astype(str).eq(str(fixture_id))].copy()
    if history.empty:
        return incoming
    history = history.sort_values("observed_at", kind="stable").drop_duplicates(identity, keep="last")
    current = set(incoming[identity].astype(str)) if identity in incoming else set()
    missing = history.loc[~history[identity].astype(str).isin(current)].copy()
    # A repeated cached response must not create a different tombstone at the same time.
    missing = missing.loc[pd.to_datetime(missing.observed_at, utc=True) < pd.Timestamp(observed_at)]
    if missing.empty:
        return incoming
    keep = {"fixture_id", "match_id", "team_id", identity, "provider_schema_version"}
    for column in missing:
        if column not in keep:
            missing[column] = None
    missing["observed_at"] = pd.Timestamp(observed_at).isoformat()
    missing["is_deleted"] = True
    if identity == "player_id":
        missing["minutes"] = 0.0
    return pd.concat([incoming, missing], ignore_index=True)


def complete_responses(frame: pd.DataFrame, revisions: pd.DataFrame, *, artifact: str, clock: str = "observed_at") -> pd.DataFrame:
    """Select the latest successful whole response for each fixture, even if empty."""
    if frame.empty:
        return frame
    if revisions.empty:
        times = pd.to_datetime(frame[clock], utc=True, errors="raise")
        selected = frame.loc[times.eq(times.groupby(frame.fixture_id).transform("max"))].copy()
    else:
        known = revisions.loc[revisions.artifact.eq(artifact)].copy()
        known["observed_at"] = pd.to_datetime(known.observed_at, utc=True, errors="raise")
        latest = known.sort_values("observed_at", kind="stable").drop_duplicates("fixture_id", keep="last").set_index("fixture_id").observed_at
        times = pd.to_datetime(frame[clock], utc=True, errors="raise")
        wanted = pd.to_datetime(frame.fixture_id.map(latest), utc=True)
        # Old archives without the response index retain their complete latest revision.
        fallback = times.groupby(frame.fixture_id).transform("max")
        selected = frame.loc[times.eq(wanted.fillna(fallback))].copy()
    if "is_deleted" in selected:
        selected = selected.loc[~selected.is_deleted.fillna(False).astype(bool)]
    return selected
