"""Fair prices, market movement, disagreement, and zero-default portfolio policy."""

import pandas as pd
import streamlit as st

from pitch_oracle_core.ui.components.freshness import render_capability


def render(context) -> None:
    st.title("Market Lab")
    st.caption(
        "Market comparison is separate from the independent forecast. Historical simulation does not promise future returns."
    )
    if not context.repository.available("odds_snapshots"):
        report = context.capabilities.get("odds", {
            "status": "unavailable",
            "coverage": 0.0,
            "source": "none",
            "message": "No point-in-time odds provider is configured for this edition.",
        })
        render_capability({"name": "Odds", **report})
        st.info(
            "Market Lab is capability-gated. Independent forecasts remain available; "
            "fair-price, movement, CLV, and portfolio views appear only with audited odds snapshots."
        )
        return
    quotes = context.repository.frame("odds_snapshots")
    if quotes.empty:
        st.info("The odds provider is configured, but no snapshots are available for this edition.")
        return
    fixture_ids = sorted(quotes.fixture_id.astype(str).unique())
    fixture_id = st.selectbox("Fixture", fixture_ids)
    selected = quotes.loc[quotes.fixture_id.astype(str) == fixture_id]
    st.dataframe(selected.sort_values("observed_at"), hide_index=True, width="stretch")
    if context.repository.available("market_assessments"):
        assessments = context.repository.frame("market_assessments")
        assessments = assessments.loc[assessments.fixture_id.astype(str) == fixture_id]
        st.subheader("Consensus and model fair prices")
        st.dataframe(assessments, hide_index=True, width="stretch")
    if context.repository.available("portfolio_backtest"):
        st.subheader("Responsible portfolio backtest")
        backtest = context.repository.frame("portfolio_backtest")
        metrics = st.columns(3)
        metrics[0].metric("Final bankroll", f"{backtest.bankroll.iloc[-1]:,.2f}")
        metrics[1].metric("Max drawdown", f"{backtest.drawdown.max():.1%}")
        metrics[2].metric("Turnover", f"{backtest.stake.sum():,.2f}")
        st.line_chart(backtest.set_index("kickoff_utc")[["bankroll"]])
        st.dataframe(backtest, hide_index=True, width="stretch")
        st.download_button(
            "Download portfolio backtest", backtest.to_csv(index=False),
            "portfolio-backtest.csv", mime="text/csv",
        )
