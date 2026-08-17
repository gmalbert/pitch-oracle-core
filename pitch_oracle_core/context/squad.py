"""Point-in-time squad availability impact with replacement shrinkage."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class PlayerAvailability:
    player_id: str
    team_id: str
    status: str
    availability_probability: float
    observed_at: str
    source: str

    def __post_init__(self) -> None:
        if not 0 <= self.availability_probability <= 1:
            raise ValueError("availability_probability must be in [0, 1]")


def availability_at_kickoff(
    availability: pd.DataFrame, fixtures: pd.DataFrame
) -> pd.DataFrame:
    """Return the latest valid pre-kickoff row per fixture and player."""
    required = {
        "fixture_id", "player_id", "team_id", "availability_probability",
        "observed_at",
    }
    missing = required.difference(availability.columns)
    if missing:
        raise ValueError(f"Missing availability columns: {sorted(missing)}")
    frame = availability.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["effective_from"] = pd.to_datetime(
        frame.get("effective_from", frame["observed_at"]), utc=True
    )
    frame["effective_to"] = pd.to_datetime(
        frame.get("effective_to", pd.Series(pd.NaT, index=frame.index)), utc=True
    )
    kickoff = fixtures[["fixture_id", "kickoff_utc"]].copy()
    kickoff["kickoff_utc"] = pd.to_datetime(kickoff["kickoff_utc"], utc=True)
    joined = frame.merge(kickoff, on="fixture_id", how="left", validate="many_to_one")
    valid = (
        (joined.observed_at < joined.kickoff_utc)
        & (joined.effective_from <= joined.kickoff_utc)
        & (joined.effective_to.isna() | (joined.kickoff_utc < joined.effective_to))
    )
    return (
        joined.loc[valid]
        .sort_values(["fixture_id", "player_id", "observed_at"])
        .drop_duplicates(["fixture_id", "player_id"], keep="last")
        .reset_index(drop=True)
    )


def squad_absence_impact(
    availability: pd.DataFrame,
    player_strength: pd.DataFrame,
    *,
    replacement_level: float = -0.25,
) -> pd.DataFrame:
    required = {"player_id", "team_id", "availability_probability"}
    if not required.issubset(availability.columns):
        raise ValueError("Availability artifact is incomplete")
    joined = availability.merge(
        player_strength[["player_id", "minutes_share", "strength_per90"]],
        on="player_id",
        how="left",
        validate="many_to_one",
    )
    joined["strength_per90"] = joined.strength_per90.fillna(replacement_level)
    joined["minutes_share"] = joined.minutes_share.fillna(0).clip(0, 1)
    joined["expected_absence"] = 1 - joined.availability_probability.clip(0, 1)
    joined["impact"] = (
        joined.expected_absence
        * joined.minutes_share
        * (joined.strength_per90 - replacement_level)
    )
    return joined.groupby("team_id", as_index=False).agg(
        expected_missing_strength=("impact", "sum"),
        players_flagged=(
            "expected_absence", lambda series: int((series > 0.25).sum())
        ),
    )
