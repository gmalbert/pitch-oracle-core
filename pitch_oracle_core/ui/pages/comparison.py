"""Two-club percentile comparison and recency-weighted meetings."""

import pandas as pd
import plotly.express as px
import streamlit as st

from pitch_oracle_core.analytics.comparison import head_to_head_context


def render(context) -> None:
    st.title("Team Comparison Studio")
    snapshots = context.repository.frame("team_snapshots")
    ranking_columns = [
        column for column in (
            "power_rank", "team_name", "elo_rating", "rating_change_week",
            "attack_l10", "defense_l10",
        ) if column in snapshots
    ]
    if ranking_columns:
        with st.expander("Dynamic power rankings", expanded=True):
            st.caption("Pre-match Elo strength with week-over-week movement.")
            st.dataframe(
                snapshots.sort_values("power_rank")[ranking_columns],
                hide_index=True,
                width="stretch",
            )
    label_column = "team_name" if "team_name" in snapshots else "team_id"
    labels = snapshots[label_column].astype(str).tolist()
    left, right = st.columns(2)
    team_a = left.selectbox("First team", labels, index=0)
    team_b = right.selectbox("Second team", labels, index=min(1, len(labels) - 1))
    if team_a == team_b:
        st.warning("Choose two different teams.")
        return
    selected = snapshots.loc[snapshots[label_column].isin([team_a, team_b])].copy()
    numeric = selected.select_dtypes("number").columns.tolist()
    metrics = [
        item for item in (
            "elo_rating", "attack_l10", "defense_l10", "points_per_match_l10",
            "clean_sheet_rate_l10", "shots_for_ewm10", "shot_quality_ewm10",
            "discipline_rate_l10", "recovery_load", "projection_expected_points",
        ) if item in numeric
    ]
    if metrics:
        percentile = snapshots[metrics].rank(pct=True)
        percentile[label_column] = snapshots[label_column]
        long = percentile.loc[percentile[label_column].isin([team_a, team_b])].melt(
            id_vars=label_column, value_vars=metrics, var_name="Metric", value_name="League percentile"
        )
        figure = px.line_polar(
            long, r="League percentile", theta="Metric", color=label_column,
            line_close=True, range_r=[0, 1],
        )
        st.plotly_chart(figure, width="stretch")
        st.dataframe(selected[[label_column, *metrics]], hide_index=True, width="stretch")
    if context.repository.available("team_events"):
        events = context.repository.frame("team_events")
        ids = selected.set_index(label_column).team_id.astype(str).to_dict()
        context_result = head_to_head_context(
            events, ids[team_a], ids[team_b],
            as_of=pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1),
        )
        meetings = context_result.get("meetings", pd.DataFrame())
        st.subheader("Head-to-head context")
        if context_result["quality"] == "insufficient":
            st.info("No recent canonical meetings; the comparison remains available.")
        else:
            st.caption(
                f"{context_result['matches']} meetings · effective n="
                f"{context_result['effective_matches']:.1f} · "
                f"sample quality {context_result['quality'].replace('_', ' ')}. "
                "Home and away meetings share the selected team's perspective."
            )
            st.dataframe(meetings.head(10), hide_index=True, width="stretch")
