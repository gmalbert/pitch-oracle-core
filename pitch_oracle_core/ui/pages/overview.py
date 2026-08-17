"""Today-first overview: pulse, race, storylines, and data freshness."""

import pandas as pd
import streamlit as st
from urllib.parse import quote

from pitch_oracle_core.ui.components.freshness import render_capability


def render(context) -> None:
    st.title(f"{context.config.display_name} intelligence")
    st.caption(f"Competition edition {context.edition_id}")
    if context.repository.available("radars"):
        radar = context.repository.frame("radars")
        st.subheader("Matchday pulse")
        columns = st.columns(3)
        for column, metric, label in (
            (columns[0], "upset_index", "Upset radar"),
            (columns[1], "draw_index", "Draw radar"),
            (columns[2], "goal_fest_percentile", "Goal fest"),
        ):
            if metric in radar and not radar.empty:
                item = radar.nlargest(1, metric).iloc[0]
                column.metric(label, f"{float(item[metric]):.0%}")
                column.caption(str(item.get("display_name", item.fixture_id)))
    elif context.repository.available("forecasts"):
        forecasts = context.repository.frame("forecasts")
        st.metric("Upcoming forecasts", len(forecasts))
    if context.repository.available("storylines"):
        st.subheader("Matchday storylines")
        for row in context.repository.frame("storylines").itertuples():
            target = quote(str(row.fixture_id), safe="")
            link_path = str(getattr(row, "link_path", "/match-center?fixture="))
            st.markdown(
                f"- **{row.storyline}:** {row.metric} = {row.value:.3f} "
                f"[Open evidence]({link_path}{target})"
            )
    if context.repository.available("season_simulations"):
        st.subheader("Race snapshot")
        projections = context.repository.frame("season_simulations")
        available = [
            column for column in ("team_name", "expected_position", "expected_points", "p_title", "p_relegation")
            if column in projections
        ]
        st.dataframe(projections[available].head(8), hide_index=True, width="stretch")
    st.subheader("Data freshness")
    if context.capabilities:
        columns = st.columns(min(3, len(context.capabilities)))
        for index, (name, report) in enumerate(context.capabilities.items()):
            with columns[index % len(columns)]:
                render_capability({"name": name.replace("_", " ").title(), **report})
    else:
        st.info("Capability reports have not been published yet.")
