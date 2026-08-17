"""One-vs-rest reliability curves with sample counts."""

import plotly.graph_objects as go
import streamlit as st
import pandas as pd


def render_reliability(curves) -> None:
    figure = go.Figure()
    figure.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect calibration",
        line=dict(color="#98a2b3", dash="dash"),
    )
    colors = {"home": "#1554a6", "draw": "#667085", "away": "#b54708"}
    for outcome, frame in curves.groupby("outcome"):
        x_column = "mean_forecast" if "mean_forecast" in frame else "forecast_mean"
        n_column = "n" if "n" in frame else "count"
        figure.add_scatter(
            x=frame[x_column],
            y=frame.observed_rate,
            mode="lines+markers",
            name=outcome.title(),
            line=dict(color=colors.get(outcome, "#344054")),
            customdata=frame[[n_column]],
            hovertemplate=(
                "Forecast %{x:.0%}<br>Observed %{y:.0%}<br>"
                "n=%{customdata[0]}<extra></extra>"
            ),
        )
    figure.update_layout(
        xaxis_title="Mean forecast",
        yaxis_title="Observed frequency",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.dataframe(curves, hide_index=True, width="stretch")
    st.download_button(
        "Download reliability data", curves.to_csv(index=False),
        "reliability.csv", mime="text/csv",
    )


def render_confidence_histogram(predictions) -> None:
    """Show OOF forecast sharpness without treating fitted probabilities as evidence."""
    required = {"p_home", "p_draw", "p_away"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Confidence histogram misses: {sorted(missing)}")
    if "prediction_source" in predictions:
        source = set(predictions.prediction_source.dropna().astype(str))
        if source != {"out_of_fold"}:
            raise ValueError("calibration diagnostics require out-of-fold predictions only")
    confidence = predictions[["p_home", "p_draw", "p_away"]].max(axis=1)
    buckets = pd.cut(confidence, bins=[0, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
    histogram = (
        pd.DataFrame({"confidence_bucket": buckets.astype(str), "confidence": confidence})
        .groupby("confidence_bucket", observed=False)
        .agg(count=("confidence", "size"), mean_confidence=("confidence", "mean"))
        .reset_index()
    )
    figure = go.Figure(go.Bar(
        x=histogram.confidence_bucket,
        y=histogram["count"],
        customdata=histogram[["mean_confidence"]],
        hovertemplate="%{x}<br>n=%{y}<br>mean=%{customdata[0]:.1%}<extra></extra>",
    ))
    figure.update_layout(
        xaxis_title="Maximum 1X2 probability", yaxis_title="Out-of-fold forecasts",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.dataframe(histogram, hide_index=True, width="stretch")
