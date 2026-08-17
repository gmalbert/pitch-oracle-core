"""Structured forecast contributions and evidence bullets."""

import pandas as pd
import plotly.express as px
import streamlit as st


def render_drivers(drivers: pd.DataFrame, outcome: str) -> None:
    selected = (
        drivers.loc[drivers["outcome"] == outcome]
        .assign(abs_contribution=lambda frame: frame["contribution"].abs())
        .nlargest(8, "abs_contribution")
        .sort_values("contribution")
    )
    if selected.empty:
        st.info("Structured forecast drivers are not available for this model.")
        return
    hover = [
        column for column in ("value", "sample_timestamp", "source")
        if column in selected.columns
    ]
    figure = px.bar(
        selected,
        x="contribution",
        y="display_name",
        orientation="h",
        color="contribution",
        color_continuous_scale=["#b42318", "#f2f4f7", "#027a48"],
        labels={"contribution": "Probability contribution", "display_name": ""},
        hover_data=hover,
    )
    figure.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    visible = [
        column for column in (
            "display_name", "contribution", "value", "sample_timestamp", "source"
        ) if column in selected
    ]
    st.dataframe(selected[visible], hide_index=True, width="stretch")
    st.download_button(
        f"Download {outcome} drivers", selected[visible].to_csv(index=False),
        f"{outcome}-forecast-drivers.csv", mime="text/csv",
    )


def evidence_bullets(drivers: pd.DataFrame, caveat: str) -> list[str]:
    strongest = drivers.assign(
        abs_value=drivers.contribution.abs()
    ).nlargest(2, "abs_value")
    bullets = [
        f"**{row.display_name}:** {row.explanation}" for row in strongest.itertuples()
    ]
    bullets.append(f"**Uncertainty:** {caveat}")
    return bullets
