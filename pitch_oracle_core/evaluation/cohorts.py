"""Contextual performance slices with minimum-sample protection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scores import score_panel


def assign_evaluation_cohorts(
    evaluations: pd.DataFrame, *, as_of: object | None = None
) -> pd.DataFrame:
    """Explode point-in-time evaluation rows into the required governance cohorts."""
    required = {"fixture_id", "kickoff_utc", "p_home", "p_draw", "p_away"}
    missing = required.difference(evaluations.columns)
    if missing:
        raise ValueError(f"Missing cohort-tag columns: {sorted(missing)}")
    frame = evaluations.copy()
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True)
    if "issued_at" in frame:
        frame["issued_at"] = pd.to_datetime(frame["issued_at"], utc=True)
        if (frame["issued_at"] >= frame["kickoff_utc"]).any():
            raise ValueError("evaluation cohorts require pre-kickoff forecasts")
    reference = pd.Timestamp(as_of) if as_of is not None else frame.kickoff_utc.max()
    if reference.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    confidence = frame[["p_home", "p_draw", "p_away"]].max(axis=1)
    confidence_decile = pd.cut(
        confidence, np.linspace(0, 1, 11), labels=False, include_lowest=True
    ).fillna(0).astype(int)
    rows: list[dict[str, object]] = []
    for position, (_, item) in enumerate(frame.iterrows()):
        tags = []
        edition = item.get("edition_id")
        if pd.notna(edition):
            tags.append(f"edition:{edition}")
        age_days = (reference - item.kickoff_utc).total_seconds() / 86_400
        for window in (90, 180, 365):
            if 0 <= age_days <= window:
                tags.append(f"rolling_{window}d")
        round_number = item.get("round_number")
        tags.append(
            "season:first_five_matchdays"
            if pd.notna(round_number) and float(round_number) <= 5
            else "season:established"
        )
        cold = str(item.get("cold_start_status", "full_history"))
        tags.append(f"history:{cold}")
        p_home, p_away = float(item.p_home), float(item.p_away)
        if p_home >= 0.55:
            tags.append("favorite:home")
        elif p_away >= 0.55:
            tags.append("favorite:away")
        else:
            tags.append("favorite:balanced")
        short_rest = bool(item.get("short_rest", False))
        tags.append("schedule:short_rest" if short_rest else "schedule:normal_rest")
        tags.append(f"phase:{item.get('phase', 'regular')}")
        tags.append(f"coverage:{item.get('coverage_status', 'full')}")
        tags.append(f"confidence_decile:{confidence_decile.iloc[position]}")
        for tag in dict.fromkeys(tags):
            rows.append({**item.to_dict(), "cohort": tag})
    return pd.DataFrame(rows)


def cohort_metrics(evaluations: pd.DataFrame, *, minimum_sample: int = 20) -> pd.DataFrame:
    required = {"cohort", "target", "p_home", "p_draw", "p_away"}
    missing = required.difference(evaluations.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")
    rows = []
    for cohort, group in evaluations.groupby("cohort", observed=True):
        probability = group[["p_home", "p_draw", "p_away"]].to_numpy()
        y = group["target"].to_numpy(dtype=int)
        if len(group) < minimum_sample:
            rows.append({"cohort": cohort, "n": len(group), "status": "insufficient_sample"})
            continue
        panel = score_panel(y, probability)
        rows.append({
            "cohort": cohort,
            "n": panel.fixtures,
            "status": "available",
            "log_loss": panel.log_loss,
            "brier": panel.brier,
            "rps": panel.rps,
            "accuracy": panel.accuracy,
            "draw_recall": panel.draw_recall,
            "standard_error_brier": float(np.std(
                np.square(probability - np.eye(3)[y]).sum(axis=1), ddof=1
            ) / np.sqrt(len(group))),
        })
    return pd.DataFrame(rows)
