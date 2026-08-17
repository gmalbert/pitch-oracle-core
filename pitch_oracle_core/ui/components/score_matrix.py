"""Scoreline heatmap and mechanically reconciled goal ladder."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def render_score_matrix(
    matrix: np.ndarray, home: str, away: str, shown_goals: int = 6
) -> None:
    full = np.asarray(matrix, dtype=float)
    values = full[: shown_goals + 1, : shown_goals + 1]
    figure = px.imshow(
        values * 100.0,
        x=[str(goal) for goal in range(values.shape[1])],
        y=[str(goal) for goal in range(values.shape[0])],
        labels={
            "x": f"{away} goals", "y": f"{home} goals", "color": "Probability %"
        },
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="Blues",
    )
    figure.update_layout(
        margin=dict(l=10, r=10, t=20, b=10), coloraxis_colorbar=dict(title="%")
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    shown_mass = values.sum()
    if shown_mass < 0.999:
        st.caption(
            f"{1 - shown_mass:.2%} probability lies outside the displayed score grid."
        )
    st.download_button(
        "Download score probabilities",
        pd.DataFrame(full).to_csv(index_label=f"{home} goals / {away} goals"),
        file_name="score-probabilities.csv",
        mime="text/csv",
    )


def goal_market_frame(markets: dict[str, float]) -> pd.DataFrame:
    rows = []
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        suffix = str(line).replace(".", "_")
        rows.append({
            "Line": line,
            "Over": markets[f"p_over_{suffix}"],
            "Under": markets[f"p_under_{suffix}"],
        })
    return pd.DataFrame(rows)
