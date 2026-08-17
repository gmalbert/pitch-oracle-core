"""Rule-aware live table with edition-specific outcome labels."""

import pandas as pd
import streamlit as st
from urllib.parse import quote

from pitch_oracle_core.competition.standings import calculate_table
from pitch_oracle_core.domain.competitions import rules_from_league_config


def render(context) -> None:
    st.title("Rule-aware Live Table")
    if context.repository.available("standings"):
        table = context.repository.frame("standings")
    else:
        fixtures = context.repository.frame("fixtures")
        required = {"home_goals", "away_goals"}
        if not required.issubset(fixtures.columns):
            st.info("Completed score artifacts are not available.")
            return
        rules_version = context.repository.descriptors.get("fixtures", {}).get(
            "rules_version"
        )
        table = calculate_table(
            fixtures,
            rules_from_league_config(context.config, version=rules_version),
        )
    st.caption(
        f"Rules version {context.repository.manifest.get('rules_version', context.edition_id)} · "
        "sanctions and phase transitions are artifact-driven."
    )
    table["team_page"] = table.team_id.astype(str).map(
        lambda value: f"/teams?team={quote(value, safe='')}"
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"team_page": st.column_config.LinkColumn("Team page", display_text="Open")},
    )
    if context.repository.available("phase_scenarios"):
        with st.expander("Split and playoff scenario explorer"):
            st.dataframe(context.repository.frame("phase_scenarios"), hide_index=True, width="stretch")
