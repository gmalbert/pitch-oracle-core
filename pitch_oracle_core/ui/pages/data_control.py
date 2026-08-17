"""Data quality control room, provenance, coverage, and artifact graph."""

import json

import pandas as pd
import streamlit as st

from pitch_oracle_core.ui.components.freshness import render_capability


def render_quality_checks(report: dict) -> None:
    checks = pd.DataFrame(report.get("checks", []))
    if checks.empty:
        st.info("No structured quality checks were published.")
        return
    blocking = checks.loc[
        (checks.severity == "blocking") & (checks.status != "passed")
    ]
    if not blocking.empty:
        st.error(f"{len(blocking)} blocking data checks failed")
    else:
        st.success("All publication-blocking data checks passed")
    available = [
        column for column in ("status", "severity", "check", "observed", "expected", "message")
        if column in checks
    ]
    display = checks[available].copy()
    for column in ("observed", "expected", "message"):
        if column in display:
            display[column] = display[column].map(
                lambda value: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else "" if value is None
                else str(value)
            )
    st.dataframe(display, hide_index=True, width="stretch")


def render(context) -> None:
    st.title("Data Control Room")
    manifest = context.repository.manifest
    st.write({
        "publish status": "valid",
        "generated": manifest.get("generated_at", manifest.get("created_at")),
        "edition": context.edition_id,
        "entity registry": manifest.get("entity_registry_version"),
    })
    if manifest.get("serving_fallback"):
        fallback = manifest["serving_fallback"]
        st.warning(
            "Serving the last fully valid same-league/same-edition artifact graph: "
            f"{fallback.get('manifest')}. Primary failure: {fallback.get('primary_error')}"
        )
    if context.repository.available("quality_report"):
        render_quality_checks(context.repository.json("quality_report"))
    if context.repository.available("provider_runs"):
        st.subheader("Provider run ledger")
        runs = context.repository.frame("provider_runs")
        st.dataframe(runs, hide_index=True, width="stretch")
        st.download_button(
            "Download provider run ledger", runs.to_csv(index=False),
            "provider-runs.csv", mime="text/csv",
        )
    st.subheader("Provider capability health")
    if context.capabilities:
        for name, report in context.capabilities.items():
            with st.container(border=True):
                render_capability({"name": name.replace("_", " ").title(), **report})
    st.subheader("Artifact dependency graph")
    artifacts = pd.DataFrame(list(context.repository.descriptors.values()))
    columns = [
        item for item in (
            "name", "path", "schema_name", "schema_version", "rows", "generated_at",
            "producer", "producer_version", "rules_version", "coverage",
            "freshness_status", "fresh_until", "dependencies", "sha256", "bytes",
        ) if item in artifacts
    ]
    st.dataframe(artifacts[columns], hide_index=True, width="stretch")
    for descriptor in context.repository.descriptors.values():
        name = descriptor["name"]
        path = context.repository.path(name)
        st.download_button(
            f"Download {name}", path.read_bytes(), file_name=path.name,
            mime=descriptor.get("media_type", "application/octet-stream"),
            key=f"download_artifact_{name}",
        )
