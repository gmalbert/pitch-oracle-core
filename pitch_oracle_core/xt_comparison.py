"""xT comparison vs pretrained model — Spearman ρ of player rankings.

Sanity check that a custom-fitted xT surface agrees with the pretrained
model on player rankings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compare_xt_rankings(
    custom_xt: pd.DataFrame,
    pretrained_xt: pd.DataFrame,
    player_col: str = "player",
    xt_col: str = "xt_added_per90",
) -> dict[str, float]:
    """Compare player rankings between custom and pretrained xT models.

    Returns Spearman ρ and the number of overlapping players.
    """
    merged = custom_xt.merge(pretrained_xt, on=player_col, suffixes=("_custom", "_pretrained"))
    if len(merged) < 5:
        return {"spearman_rho": float("nan"), "n_players": len(merged)}
    rho, p_value = spearmanr(merged[f"{xt_col}_custom"], merged[f"{xt_col}_pretrained"])
    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n_players": len(merged),
    }
