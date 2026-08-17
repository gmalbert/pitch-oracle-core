"""Searchable append-only forecast issuance history."""

import pandas as pd
import streamlit as st


def render(context) -> None:
    st.title("Prediction History")
    frame = context.repository.frame("forecast_ledger")
    requested = st.query_params.get("fixture", "")
    if isinstance(requested, list):
        requested = requested[0] if requested else ""
    query = st.text_input(
        "Search fixture, model, revision, result, or score", value=str(requested)
    )
    if query:
        searchable = [
            column for column in (
                "fixture_id", "model_id", "revision_label", "result_status",
                "actual_outcome", "actual_home_goals", "actual_away_goals",
            ) if column in frame
        ]
        matches = pd.Series(False, index=frame.index)
        for column in searchable:
            matches |= frame[column].astype(str).str.contains(query, case=False, na=False)
        frame = frame.loc[matches]
    model_ids = sorted(frame.model_id.dropna().unique()) if "model_id" in frame else []
    selected = st.multiselect("Model", model_ids, default=model_ids)
    if model_ids:
        frame = frame.loc[frame.model_id.isin(selected)]
    st.dataframe(frame.sort_values("issued_at", ascending=False), hide_index=True, width="stretch")
    st.download_button("Download immutable ledger", frame.to_csv(index=False), "forecast-ledger.csv")
