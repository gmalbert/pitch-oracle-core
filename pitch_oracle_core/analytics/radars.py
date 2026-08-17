"""Upset, draw, high-event, and low-event fixture watchlists."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_entropy(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=float)
    values = values[values > 0]
    return float(-(values * np.log(values)).sum())


def build_fixture_radars(
    forecasts: pd.DataFrame,
    *,
    minimum_leader_stability: float = 0.70,
) -> pd.DataFrame:
    required = {
        "fixture_id", "p_home", "p_draw", "p_away", "expected_total_goals"
    }
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Missing radar columns: {sorted(missing)}")
    frame = forecasts.copy()
    if "home_strength_probability" not in frame:
        frame["home_strength_probability"] = frame["p_home"]
    if "away_strength_probability" not in frame:
        frame["away_strength_probability"] = frame["p_away"]
    weaker_model = np.where(
        frame["home_strength_probability"] < frame["away_strength_probability"],
        frame["p_home"],
        frame["p_away"],
    )
    weaker_baseline = np.minimum(
        frame["home_strength_probability"], frame["away_strength_probability"]
    )
    frame["upset_index_raw"] = weaker_model - weaker_baseline
    stability_source = (
        frame["leader_stability"]
        if "leader_stability" in frame
        else pd.Series(1.0, index=frame.index)
    )
    stability = pd.to_numeric(stability_source, errors="coerce").fillna(0.0)
    widths = []
    for outcome in ("home", "draw", "away"):
        lower, upper = f"p_{outcome}_lower80", f"p_{outcome}_upper80"
        if lower in frame and upper in frame:
            widths.append(frame[upper] - frame[lower])
    maximum_width = pd.concat(widths, axis=1).max(axis=1) if widths else pd.Series(0.0, index=frame.index)
    frame["uncertainty_passed"] = (
        (stability >= minimum_leader_stability) & (maximum_width <= 0.25)
    )
    frame["upset_index"] = frame["upset_index_raw"].where(frame["uncertainty_passed"])
    parity = 1 - abs(frame["p_home"] - frame["p_away"])
    low_score = 1 / (1 + frame["expected_total_goals"].clip(lower=0))
    calibrated_draw = pd.to_numeric(
        frame.get("calibrated_p_draw", frame["p_draw"]), errors="coerce"
    ).fillna(frame["p_draw"])
    frame["score_matrix_p_draw"] = frame["p_draw"]
    frame["calibrated_draw_probability"] = calibrated_draw
    frame["draw_calibration_gap"] = calibrated_draw - frame["p_draw"]
    frame["draw_index"] = calibrated_draw * 0.40 + frame["p_draw"] * 0.20 + parity * 0.25 + low_score * 0.15
    if "p_btts_yes" not in frame:
        frame["p_btts_yes"] = np.nan
    event_inputs = pd.DataFrame(index=frame.index)
    event_inputs["expected_total_goals"] = frame["expected_total_goals"].rank(pct=True)
    event_inputs["btts"] = frame["p_btts_yes"].fillna(frame["p_btts_yes"].median()).fillna(0.5).rank(pct=True)
    for source in ("score_entropy", "shots_tempo", "style_matchup"):
        if source in frame:
            event_inputs[source] = pd.to_numeric(frame[source], errors="coerce").rank(pct=True).fillna(0.5)
    frame["goal_event_score"] = event_inputs.mean(axis=1)
    percentile = frame["goal_event_score"].rank(pct=True, method="average")
    frame["goal_fest_percentile"] = percentile
    frame["low_block_percentile"] = 1 - percentile
    if "cold_start" not in frame:
        frame["cold_start"] = "full"
    frame["cold_start_label"] = frame.get(
        "cold_start_label", frame["cold_start"].astype(str).str.replace("_", " ").str.title()
    )
    return frame
