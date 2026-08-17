"""Rule-labelled season projection tables and position heatmaps."""

import pandas as pd
import plotly.express as px
import streamlit as st
from urllib.parse import quote


def render_projection_table(
    projections: pd.DataFrame, outcome_columns: list[str]
) -> None:
    projections = projections.copy()
    if "team_id" in projections:
        projections["team_page"] = projections.team_id.astype(str).map(
            lambda value: f"/teams?team={quote(value, safe='')}"
        )
    available = [
        column for column in [
            "current_position", "team_name", "team_page", "current_points", "expected_points",
            "expected_position", *outcome_columns,
        ] if column in projections.columns
    ]
    display = projections[available].sort_values("expected_position")
    column_config = {
        "current_position": st.column_config.NumberColumn("Now", format="%d"),
        "team_name": st.column_config.TextColumn("Team", pinned=True),
        "current_points": st.column_config.NumberColumn("Pts", format="%d"),
        "expected_points": st.column_config.NumberColumn("Expected pts", format="%.1f"),
        "expected_position": st.column_config.NumberColumn("Expected rank", format="%.1f"),
        "team_page": st.column_config.LinkColumn("Team page", display_text="Open"),
    }
    column_config.update({
        column: st.column_config.ProgressColumn(
            column.replace("p_", "").replace("_", " ").title(),
            min_value=0.0,
            max_value=1.0,
            format="percent",
        ) for column in outcome_columns if column in projections.columns
    })
    st.dataframe(
        display, hide_index=True, width="stretch", column_config=column_config
    )


def render_position_distribution(position_probability: pd.DataFrame) -> None:
    required = {"team_name", "position", "probability"}
    missing = required.difference(position_probability.columns)
    if missing:
        raise ValueError(f"Position distribution misses: {sorted(missing)}")
    frame = position_probability.pivot(
        index="team_name", columns="position", values="probability"
    ).fillna(0.0)
    figure = px.imshow(
        frame * 100,
        labels={"x": "Final position", "y": "", "color": "Probability %"},
        aspect="auto",
        color_continuous_scale="Blues",
    )
    figure.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.dataframe(frame, width="stretch")
    st.download_button(
        "Download finishing-position probabilities", frame.to_csv(),
        "position-probabilities.csv", mime="text/csv",
    )
