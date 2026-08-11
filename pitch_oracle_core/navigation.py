"""Shared native Streamlit navigation for all Pitch Oracle league consumers."""

from __future__ import annotations

from os import environ, path

import streamlit as st

from .config import LeagueConfig
from .footer import render_footer
from .ui_pages import (
    render_model_lab,
    render_overview,
    render_predictions,
    render_raw_data,
    render_standings,
    render_statistics,
    render_team_deep_dive,
)


def render_sidebar_branding(config: LeagueConfig, *, show_logo: bool = True) -> None:
    """Render shared, non-navigation sidebar content."""
    data_dir = environ.get("PITCH_ORACLE_DATA_DIR", "data_files")
    logo = path.join(data_dir, "logo_no_words.png")
    if show_logo and path.exists(logo):
        st.sidebar.image(logo, width=150)
    elif show_logo:
        st.sidebar.markdown("## ⚽ Pitch Oracle")

    st.sidebar.divider()


def build_navigation(config: LeagueConfig):
    """Build the sidebar navigation with explicit icons and ASCII-safe pages."""
    def with_footer(page):
        def wrapped_page() -> None:
            page()
            render_footer()
        return wrapped_page

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
                    with_footer(overview_page),
                    title="Overview",
                    icon="🏠",
                    url_path="overview",
                    default=True,
                ),
            ],
            "Match Center": [
                st.Page(with_footer(predictions_page), title="Predictions", icon="🎯", url_path="predictions"),
                st.Page(with_footer(standings_page), title="Standings", icon="🏆", url_path="standings"),
                st.Page(
                    with_footer(team_deep_dive_page),
                    title="Team Deep Dive",
                    icon="🔎",
                    url_path="team-deep-dive",
                ),
            ],
            "Analysis": [
                st.Page(with_footer(statistics_page), title="Statistics", icon="📊", url_path="statistics"),
                st.Page(with_footer(model_lab_page), title="Model Lab", icon="🧠", url_path="model-lab"),
                st.Page(with_footer(raw_data_page), title="Raw Data", icon="🗃️", url_path="raw-data"),
            ],
        }
    )
