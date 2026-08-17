"""Champion/challenger deployment, proper scores, reliability, cohorts, and drift."""

import pandas as pd
import streamlit as st

from pitch_oracle_core.evaluation.scores import score_panel
from pitch_oracle_core.evaluation.calibration import expected_calibration_error
from pitch_oracle_core.ui.components.reliability import (
    render_confidence_histogram,
    render_reliability,
)


def render(context) -> None:
    st.title("Model Lab")
    registry = context.repository.json("model_registry")
    tabs = st.tabs(["Deployment", "Performance", "Calibration", "Cohorts & drift"])
    with tabs[0]:
        st.metric("Production model", registry.get("production_model_id", "Unknown"))
        gate = registry.get("release_gate", {})
        st.write({
            "status": gate.get("status"),
            "reason": gate.get("reason"),
            "baseline": gate.get("baseline_model_id", gate.get("champion_model_id")),
        })
        st.dataframe(pd.DataFrame(registry.get("models", [])), hide_index=True, width="stretch")
    with tabs[1]:
        if context.repository.available("evaluation_predictions"):
            rows = context.repository.frame("evaluation_predictions")
            target = "actual_outcome" if "actual_outcome" in rows else "target"
            reports = []
            for model_id, group in rows.groupby("model_id"):
                panel = score_panel(
                    group[target].to_numpy(dtype=int),
                    group[["p_home", "p_draw", "p_away"]].to_numpy(),
                )
                reports.append({"model_id": model_id, **panel.__dict__})
            st.dataframe(pd.DataFrame(reports), hide_index=True, width="stretch")
            score_frame = pd.DataFrame(reports).sort_values(
                ["log_loss", "brier"], kind="stable"
            )
            reproduced = str(score_frame.iloc[0].model_id)
            production = str(registry.get("production_model_id"))
            if reproduced == production:
                st.success("Persisted out-of-time rows reproduce the production selection.")
            else:
                st.error(
                    f"Registry selects {production}, but persisted rows select {reproduced}."
                )
        else:
            st.info("Rolling-origin evaluation rows are unavailable.")
    with tabs[2]:
        if context.repository.available("calibration"):
            curves = context.repository.frame("calibration")
            render_reliability(curves)
            if context.repository.available("evaluation_predictions"):
                rows = context.repository.frame("evaluation_predictions")
                if "prediction_source" in rows and set(rows.prediction_source.astype(str)) != {"out_of_fold"}:
                    st.error("Calibration explorer is blocked: non-OOF probabilities are present.")
                else:
                    st.subheader("Confidence histogram")
                    render_confidence_histogram(rows)
                    summaries = []
                    grouping = [column for column in ("model_id", "edition_id") if column in rows]
                    for keys, group in rows.groupby(grouping, observed=True):
                        keys = keys if isinstance(keys, tuple) else (keys,)
                        summary = dict(zip(grouping, keys))
                        target = group.actual_outcome.to_numpy(dtype=int)
                        probability = group[["p_home", "p_draw", "p_away"]].to_numpy()
                        summary["brier"] = float(score_panel(target, probability).brier)
                        summary["ece"] = max(
                            expected_calibration_error(probability[:, index], target == index)
                            for index in range(3)
                        )
                        summary["n"] = len(group)
                        summaries.append(summary)
                    st.subheader("ECE and Brier by season")
                    st.dataframe(pd.DataFrame(summaries), hide_index=True, width="stretch")
                    binary_summaries = []
                    for market, probability_column, target_column in (
                        ("Over 2.5", "p_over_2_5", "actual_over_2_5"),
                        ("BTTS", "p_btts_yes", "actual_btts_yes"),
                    ):
                        if {probability_column, target_column}.issubset(rows.columns):
                            probability = rows[probability_column].to_numpy(dtype=float)
                            target = rows[target_column].to_numpy(dtype=int)
                            binary_summaries.append({
                                "market": market,
                                "brier": float(((probability - target) ** 2).mean()),
                                "ece": expected_calibration_error(probability, target),
                                "n": len(rows),
                                "source": "out_of_fold",
                            })
                    if binary_summaries:
                        st.subheader("Goal-market calibration")
                        st.dataframe(
                            pd.DataFrame(binary_summaries), hide_index=True, width="stretch"
                        )
        else:
            st.info("Calibration curves are unavailable.")
    with tabs[3]:
        if context.repository.available("cohort_metrics"):
            st.subheader("Cohort performance")
            st.dataframe(context.repository.frame("cohort_metrics"), hide_index=True, width="stretch")
        if context.repository.available("drift_report"):
            st.subheader("Forecast drift")
            st.dataframe(context.repository.frame("drift_report"), hide_index=True, width="stretch")
