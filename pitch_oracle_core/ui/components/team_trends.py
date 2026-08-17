"""Team form fingerprints with opponent/venue context."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_team_trend(
    events: pd.DataFrame, team_name: str, metric: str, label: str
) -> None:
    frame = events.sort_values("kickoff_utc").copy()
    if metric not in frame.columns:
        st.info(f"{label} is not available for this team.")
        return
    frame["rolling"] = frame[metric].ewm(halflife=5, min_periods=1).mean()
    figure = go.Figure()
    colors = frame.get("result", pd.Series("D", index=frame.index)).map(
        {"W": "#027a48", "D": "#667085", "L": "#b42318"}
    )
    custom_columns = [
        column for column in ("opponent_name", "venue_role", "score")
        if column in frame.columns
    ]
    figure.add_scatter(
        x=frame.kickoff_utc,
        y=frame[metric],
        mode="markers",
        marker_color=colors,
        name="Match",
        customdata=frame[custom_columns] if custom_columns else None,
    )
    figure.add_scatter(
        x=frame.kickoff_utc,
        y=frame["rolling"],
        mode="lines",
        name="EW trend",
        line=dict(color="#1554a6", width=3),
    )
    figure.update_layout(
        title=f"{team_name} · {label}",
        yaxis_title=label,
        margin=dict(l=10, r=10, t=45, b=10),
        legend_orientation="h",
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    alternative = ["kickoff_utc", metric, "rolling", *custom_columns]
    st.dataframe(frame[alternative], hide_index=True, width="stretch")
    st.download_button(
        f"Download {label} trend", frame[alternative].to_csv(index=False),
        f"{metric}-trend.csv", mime="text/csv",
    )
