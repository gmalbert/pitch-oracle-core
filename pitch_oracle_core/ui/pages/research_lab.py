"""Frozen-experiment research scorecard and decision record."""

import pandas as pd
import streamlit as st

from pitch_oracle_core.ui.components.reliability import render_reliability


def render_research_lab(
    experiments: pd.DataFrame,
    metric_rows: pd.DataFrame,
    calibration_rows: pd.DataFrame,
) -> None:
    st.title("Forecast Research Lab")
    st.caption(
        "All results are rolling-origin forecasts. Betting simulation is a separate decision-policy evaluation and does not prove future returns."
    )
    experiment_id = st.selectbox("Frozen experiment", experiments.experiment_id.tolist())
    experiment = experiments.loc[experiments.experiment_id == experiment_id].iloc[0]
    st.write({
        "hypothesis": experiment.hypothesis,
        "created": experiment.get("created_at"),
        "forward test": experiment.get("forward_test_range"),
        "family test": experiment.get("family_test"),
        "status": experiment.get("status"),
    })
    metrics = metric_rows.loc[metric_rows.experiment_id == experiment_id]
    curves = calibration_rows.loc[calibration_rows.experiment_id == experiment_id]
    tabs = st.tabs([
        "Scorecard", "Calibration", "Residuals", "Paired & cohorts",
        "Market benchmark", "Operations & graveyard", "Decision",
    ])
    with tabs[0]:
        st.subheader("Probability scorecard")
        st.dataframe(metrics, hide_index=True, width="stretch")
    with tabs[1]:
        st.subheader("Calibration and sharpness")
        st.caption(
            "Reliability is paired with sample counts; a low-information prior is not treated as sharp."
        )
        if not curves.empty:
            render_reliability(curves)
    with tabs[2]:
        columns = [
            column for column in (
                "model_id", "dispersion_residual", "zero_residual",
                "diagonal_residual", "tail_residual", "fit_failures",
            ) if column in metrics
        ]
        st.dataframe(metrics[columns] if columns else metrics, hide_index=True, width="stretch")
    with tabs[3]:
        columns = [
            column for column in (
                "model_id", "paired_delta", "lower_95", "upper_95",
                "family_test_p_value", "cohort", "fixtures",
            ) if column in metrics
        ]
        st.dataframe(metrics[columns] if columns else metrics, hide_index=True, width="stretch")
    with tabs[4]:
        columns = [
            column for column in (
                "model_id", "track", "market_log_loss_delta", "clv",
                "devig_method", "odds_coverage",
            ) if column in metrics
        ]
        st.dataframe(metrics[columns] if columns else metrics, hide_index=True, width="stretch")
    with tabs[5]:
        columns = [
            column for column in (
                "model_id", "fit_seconds", "inference_ms", "artifact_bytes",
                "fit_failures", "fallback_rate", "status", "rejection_reason",
            ) if column in metrics
        ]
        st.dataframe(metrics[columns] if columns else metrics, hide_index=True, width="stretch")
        rejected = experiments.loc[experiments.status.astype(str).str.contains(
            "reject|defer", case=False, regex=True
        )]
        if not rejected.empty:
            st.subheader("Experiment graveyard")
            st.dataframe(rejected, hide_index=True, width="stretch")
    with tabs[6]:
        st.subheader("Decision record")
        st.markdown(str(experiment.get("decision_markdown", "No decision recorded.")))


def render(context) -> None:
    render_research_lab(
        context.repository.frame("research_experiments"),
        context.repository.frame("research_metrics"),
        context.repository.frame("research_calibration"),
    )
