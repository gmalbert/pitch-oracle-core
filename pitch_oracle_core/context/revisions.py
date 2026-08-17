"""Immutable forecast revision labels and deltas."""

from __future__ import annotations

import pandas as pd

REVISION_LABELS = (
    (168, "initial"),
    (24, "24_hour"),
    (2, "lineup"),
    (0, "closing"),
)


def revision_label(kickoff_utc, issued_at) -> str:
    hours = (kickoff_utc - issued_at).total_seconds() / 3600
    for threshold, label in REVISION_LABELS:
        if hours >= threshold:
            return label
    return "late_or_live"


def forecast_revision_deltas(ledger: pd.DataFrame) -> pd.DataFrame:
    frame = ledger.copy()
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True)
    frame["issued_at"] = pd.to_datetime(frame["issued_at"], utc=True)
    if (frame["issued_at"] >= frame["kickoff_utc"]).any():
        raise ValueError("pre-match forecast ledger contains a late/live issue")
    immutable_key = ["fixture_id", "issued_at", "model_id"]
    available_keys = [key for key in immutable_key if key in frame.columns]
    if frame.duplicated(available_keys).any():
        raise ValueError("forecast revision keys must be immutable and unique")
    frame = frame.sort_values(["fixture_id", "issued_at"])
    for column in (
        "p_home", "p_draw", "p_away", "expected_home_goals", "expected_away_goals"
    ):
        if column in frame.columns:
            frame[f"delta_{column}"] = frame.groupby("fixture_id")[column].diff()
    frame["revision_label"] = [
        revision_label(kickoff, issued)
        for kickoff, issued in zip(frame["kickoff_utc"], frame["issued_at"])
    ]
    return frame
