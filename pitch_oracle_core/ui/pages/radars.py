"""Ranked fixture watchlists for upsets, draws, and event tempo."""

import streamlit as st


def render(context) -> None:
    st.title("Fixture Radars")
    frame = context.repository.frame("radars")
    tabs = st.tabs(["Upsets", "Draws", "Goal fest", "Low block"])
    definitions = (
        (tabs[0], "upset_index", False),
        (tabs[1], "draw_index", False),
        (tabs[2], "goal_fest_percentile", False),
        (tabs[3], "low_block_percentile", False),
    )
    for tab, metric, ascending in definitions:
        with tab:
            st.caption("League-relative ranking; cold starts remain visibly labelled.")
            display = frame
            if metric == "upset_index" and "uncertainty_passed" in frame:
                display = frame.loc[frame.uncertainty_passed.astype(bool)]
                st.caption("Upsets are filtered by interval width and leader stability.")
            st.dataframe(
                display.sort_values(metric, ascending=ascending),
                hide_index=True,
                width="stretch",
            )
