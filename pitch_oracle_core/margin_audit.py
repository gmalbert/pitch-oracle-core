"""Per-bookmaker margin audit — quarterly report on which books are sharp.

Tracks the overround (margin) each bookmaker applies, helping identify
which books offer the tightest prices.
"""

from __future__ import annotations

import pandas as pd
from penaltyblog.implied import ImpliedMethod, calculate_implied


def compute_bookmaker_margins(
    odds: pd.DataFrame,
    *,
    bookmaker_col: str = "bookmaker",
    home_col: str = "home",
    draw_col: str = "draw",
    away_col: str = "away",
) -> pd.DataFrame:
    """Compute the overround margin for each bookmaker-market observation."""
    margins = []
    for _, row in odds.iterrows():
        try:
            result = calculate_implied(
                [row[home_col], row[draw_col], row[away_col]],
                method=ImpliedMethod.MULTIPLICATIVE,
            )
            margins.append({
                "bookmaker": row[bookmaker_col],
                "match_id": row.get("match_id", ""),
                "margin": result.margin,
            })
        except (ValueError, KeyError):
            continue
    return pd.DataFrame(margins)


def margin_audit_summary(margins: pd.DataFrame) -> pd.DataFrame:
    """Aggregate margin statistics per bookmaker."""
    return margins.groupby("bookmaker").agg(
        margin_mean=("margin", "mean"),
        margin_median=("margin", "median"),
        margin_p95=("margin", lambda s: s.quantile(0.95)),
        n_markets=("margin", "count"),
    ).sort_values("margin_mean")
