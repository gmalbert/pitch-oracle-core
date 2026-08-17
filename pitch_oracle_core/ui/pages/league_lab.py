"""League trends, balance, storylines, and cross-league comparisons."""

import plotly.express as px
import streamlit as st
from urllib.parse import quote


def render(context) -> None:
    st.title("League Laboratory")
    if context.repository.available("league_trends"):
        trends = context.repository.frame("league_trends")
        candidates = [column for column in trends if column.startswith("rolling_")]
        metric = st.selectbox("Trend", candidates)
        figure = px.line(
            trends, x="kickoff_utc", y=metric,
            labels={metric: metric.replace("rolling_", "").replace("_", " ").title()},
        )
        st.plotly_chart(figure, width="stretch")
        st.caption("Rolling window and sample size are published in the source table.")
        st.dataframe(trends[["kickoff_utc", metric, "sample_n"]], hide_index=True, width="stretch")
        st.download_button(
            "Download league trend", trends.to_csv(index=False),
            "league-trends.csv", mime="text/csv",
        )
    if context.repository.available("competitive_balance"):
        st.subheader("Competitive balance")
        balance = context.repository.frame("competitive_balance")
        st.dataframe(balance, hide_index=True, width="stretch")
    if context.repository.available("cross_league"):
        st.subheader("Aligned cross-league comparison")
        st.dataframe(context.repository.frame("cross_league"), hide_index=True, width="stretch")
    if context.repository.available("storylines"):
        st.subheader("Deterministic storylines")
        stories = context.repository.frame("storylines").copy()
        stories["fixture_page"] = stories.apply(
            lambda row: str(row.get("link_path", "/match-center?fixture="))
            + quote(str(row.fixture_id), safe=""),
            axis=1,
        )
        st.dataframe(
            stories, hide_index=True, width="stretch",
            column_config={"fixture_page": st.column_config.LinkColumn("Fixture", display_text="Open")},
        )
