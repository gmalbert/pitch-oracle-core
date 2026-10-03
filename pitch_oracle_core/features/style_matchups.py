"""Small symmetric, hypothesis-driven interactions over already eligible state."""

import numpy as np
import pandas as pd


def _col(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def add_style_matchups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for metric in ("field_tilt", "direct_speed"):
        result[f"{metric}_gap"] = _col(frame, f"home_{metric}_ewm5") - _col(frame, f"away_{metric}_ewm5")
    for side, opponent in (("home", "away"), ("away", "home")):
        result[f"{side}_press_vs_{opponent}_buildup"] = _col(frame, f"{side}_high_turnovers_ewm5") * _col(frame, f"{opponent}_passes_per_sequence_ewm5")
        result[f"{side}_territory_pressure"] = _col(frame, f"{side}_box_entries_ewm5") * _col(frame, f"{opponent}_ppda_ewm5")
        result[f"{side}_creation_quality"] = _col(frame, f"{side}_xt_for_ewm5") * _col(frame, f"{side}_xg_per_shot_ewm5")
    return result
