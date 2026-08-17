"""Source provenance and freshness cards."""

from datetime import datetime, timezone
import pandas as pd
import streamlit as st

from pitch_oracle_core.ui.formatters import freshness


def render_capability(report: dict) -> None:
    observed = pd.to_datetime(report.get("observed_at"), utc=True, errors="coerce")
    if pd.isna(observed):
        age = float("inf")
        updated = "No successful observation"
    else:
        age = (datetime.now(timezone.utc) - observed.to_pydatetime()).total_seconds() / 60
        updated = observed.strftime("%Y-%m-%d %H:%M UTC")
    label, _ = freshness(age)
    st.metric(report.get("name", "Source"), str(report.get("status", "unknown")).title())
    st.caption(
        f"{label} · coverage {float(report.get('coverage', report.get('coverage_fraction', 0))):.0%} "
        f"· {updated} · {report.get('source', report.get('provider', 'unknown'))}"
    )
    if report.get("message") or report.get("reason"):
        st.caption(str(report.get("message") or report.get("reason")))
