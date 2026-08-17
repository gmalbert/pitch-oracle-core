"""Season projections, race outcomes, positions, targets, and stakes."""

import streamlit as st

from pitch_oracle_core.ui.components.projection_table import (
    render_position_distribution,
    render_projection_table,
)


def render(context) -> None:
    st.title("Season Command Center")
    projections = context.repository.frame("season_simulations")
    outcome_columns = [column for column in projections if column.startswith("p_")]
    render_projection_table(projections, outcome_columns)
    if context.repository.available("position_probabilities"):
        st.subheader("Finishing-position distribution")
        render_position_distribution(context.repository.frame("position_probabilities"))
    if context.repository.available("points_targets"):
        st.subheader("Points target ranges")
        st.caption("Targets are simulation distributions, not false single cutoffs.")
        st.dataframe(context.repository.frame("points_targets"), hide_index=True, width="stretch")
    if context.repository.available("match_stakes"):
        st.subheader("Matchday stakes")
        st.dataframe(
            context.repository.frame("match_stakes").sort_values("index", ascending=False),
            hide_index=True,
            width="stretch",
        )
