"""Streamlit page: Today's Bets — value bets and arbitrage opportunities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def render(context):
    """Render the Today's Bets page."""
    st.title("Today's Bets")
    st.caption("Value bets and arbitrage opportunities from the daily scan")

    data_dir = Path(context.config.data_dir_name)
    scan_path = data_dir / "value" / "daily_scan.parquet"

    if not scan_path.exists():
        st.info("No value scan available. Run `python scripts/value/daily_value_scan.py` first.")
        return

    scan = pd.read_parquet(scan_path)
    if scan.empty:
        st.info("No qualifying opportunities found today.")
        return

    # Split by type
    arbs = scan[scan["type"] == "arbitrage"]
    values = scan[scan["type"] == "value"]

    if not arbs.empty:
        st.subheader("Arbitrage Opportunities")
        st.dataframe(arbs[["game", "return"]], use_container_width=True)

    if not values.empty:
        st.subheader("Value Bets")
        # Sort by expected value
        values = values.sort_values("ev", ascending=False)
        st.dataframe(
            values[["game", "outcome", "edge", "ev", "tier", "kelly_stake"]],
            use_container_width=True,
        )

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value Bets", len(values))
        col2.metric("Best Edge", f"{values['edge'].max():.1%}")
        col3.metric("Best EV", f"{values['ev'].max():.1%}")
