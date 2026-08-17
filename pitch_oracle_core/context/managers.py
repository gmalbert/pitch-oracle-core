"""Manager identity at kickoff and shrinkage-aware change effects."""

from __future__ import annotations

import pandas as pd


def attach_manager_at_kickoff(
    matches: pd.DataFrame, tenures: pd.DataFrame
) -> pd.DataFrame:
    required = {"team_id", "manager_id", "effective_from"}
    missing = required.difference(tenures.columns)
    if missing:
        raise ValueError(f"Missing manager-tenure columns: {sorted(missing)}")
    match_frame, tenure_frame = matches.copy(), tenures.copy()
    match_frame["kickoff_utc"] = pd.to_datetime(match_frame["kickoff_utc"], utc=True)
    tenure_frame["effective_from"] = pd.to_datetime(
        tenure_frame["effective_from"], utc=True
    )
    if "effective_to" in tenure_frame:
        tenure_frame["effective_to"] = pd.to_datetime(
            tenure_frame["effective_to"], utc=True
        )
    else:
        tenure_frame["effective_to"] = pd.NaT
    if "observed_at" in tenure_frame:
        tenure_frame["observed_at"] = pd.to_datetime(
            tenure_frame["observed_at"], utc=True
        )
    else:
        tenure_frame["observed_at"] = tenure_frame["effective_from"]
    rows = []
    for side in ("home", "away"):
        left = match_frame[["fixture_id", "kickoff_utc", f"{side}_team_id"]].rename(
            columns={f"{side}_team_id": "team_id"}
        )
        merged = left.merge(tenure_frame, on="team_id", how="left")
        valid = (
            (merged["observed_at"] < merged["kickoff_utc"])
            & (merged["effective_from"] <= merged["kickoff_utc"])
            & (merged["effective_to"].isna() | (merged["kickoff_utc"] < merged["effective_to"]))
        )
        selected = (
            merged.loc[valid]
            .sort_values(["fixture_id", "observed_at", "effective_from"])
            .drop_duplicates("fixture_id", keep="last")
        )
        complete = left[["fixture_id"]].merge(
            selected[["fixture_id", "manager_id"]], on="fixture_id", how="left"
        )
        rows.append(complete.rename(
            columns={"manager_id": f"{side}_manager_id"}
        ))
    return rows[0].merge(rows[1], on="fixture_id", validate="one_to_one")


def shrunk_manager_effect(
    residual_points: pd.Series,
    *,
    prior_mean: float = 0.0,
    prior_matches: float = 12.0,
) -> float:
    if prior_matches <= 0:
        raise ValueError("prior_matches must be positive")
    values = residual_points.dropna().to_numpy(dtype=float)
    return float(
        (values.sum() + prior_mean * prior_matches) / (len(values) + prior_matches)
    )
