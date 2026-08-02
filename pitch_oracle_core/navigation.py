"""Shared native Streamlit navigation for all Pitch Oracle league consumers."""

from __future__ import annotations

from os import path

import streamlit as st

from .config import LeagueConfig
from .ui_pages import (
    render_model_lab,
    render_overview,
    render_predictions,
    render_raw_data,
    render_standings,
    render_statistics,
    render_team_deep_dive,
)


def render_sidebar_branding(config: LeagueConfig) -> None:
    """Render shared, non-navigation sidebar content."""
    logo = path.join("data_files", "logo.png")
    if path.exists(logo):
        st.sidebar.image(logo, width=150)
    else:
        st.sidebar.markdown("## ⚽ Pitch Oracle")


def build_navigation(config: LeagueConfig):
    """Build the sidebar navigation with explicit icons and ASCII-safe pages."""
    def overview_page() -> None:
        render_overview(config)

    def predictions_page() -> None:
        render_predictions(config)

    def standings_page() -> None:
        render_standings(config)

    def team_deep_dive_page() -> None:
        render_team_deep_dive(config)

    def statistics_page() -> None:
        render_statistics(config)

    def model_lab_page() -> None:
        render_model_lab(config)

    def raw_data_page() -> None:
        render_raw_data(config)

    return st.navigation(
        {
            "": [
                st.Page(
                    overview_page,
                    title="Overview",
                    icon="🏠",
                    url_path="overview",
                    default=True,
                ),
            ],
            "Match Center": [
                st.Page(predictions_page, title="Predictions", icon="🎯", url_path="predictions"),
                st.Page(standings_page, title="Standings", icon="🏆", url_path="standings"),
                st.Page(
                    team_deep_dive_page,
                    title="Team Deep Dive",
                    icon="🔎",
                    url_path="team-deep-dive",
                ),
            ],
            "Analysis": [
                st.Page(statistics_page, title="Statistics", icon="📊", url_path="statistics"),
                st.Page(model_lab_page, title="Model Lab", icon="🧠", url_path="model-lab"),
                st.Page(raw_data_page, title="Raw Data", icon="🗃️", url_path="raw-data"),
            ],
        }
    )
