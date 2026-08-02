"""Shared visual theme for Pitch Oracle Streamlit consumers."""

from __future__ import annotations

import streamlit as st

from .config import LeagueConfig


def apply_theme(config: LeagueConfig) -> None:
    """Apply the shared Pitch Oracle visual system.

    Consumers provide the league identity; the core owns the common visual
    language so every league deployment feels like the same product.
    """
    primary = config.theme.primary
    primary_dark = config.theme.primary_dark
    sidebar = config.theme.sidebar
    page = config.theme.page
    border = config.theme.border
    muted = config.theme.muted

    st.markdown(
        f"""
        <style>
        :root {{
            --pitch-primary: {primary};
            --pitch-primary-dark: {primary_dark};
            --pitch-sidebar: {sidebar};
            --pitch-page: {page};
            --pitch-border: {border};
            --pitch-muted: {muted};
        }}

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {{
            background: var(--pitch-page);
        }}

        [data-testid="stSidebar"] {{
            background: var(--pitch-sidebar);
            border-right: 1px solid var(--pitch-border);
        }}

        [data-testid="stSidebarContent"] {{
            padding-top: 1.25rem;
        }}

        [data-testid="stMainBlockContainer"] {{
            max-width: 1220px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: var(--pitch-primary-dark);
            letter-spacing: -0.02em;
        }}

        h1 {{
            margin-bottom: 0.35rem;
        }}

        [data-testid="stCaptionContainer"] {{
            color: var(--pitch-muted);
        }}

        /* Preserve the complete logo/artwork in responsive image containers. */
        [data-testid="stImage"] {{
            overflow: visible;
            /* The logo's ball reaches the edge of its PNG canvas.  A small
               inset prevents it from being clipped when rendered in the
               overview header's constrained column. */
            box-sizing: border-box;
            padding: 0.35rem;
        }}

        [data-testid="stImage"] img {{
            height: auto;
            max-width: 100%;
            object-fit: contain;
            object-position: center;
        }}

        [data-testid="stMetric"] {{
            background: #f8fafc;
            border: 1px solid var(--pitch-border);
            border-radius: 10px;
            padding: 0.85rem 1rem;
        }}

        [data-testid="stMetricValue"] {{
            color: var(--pitch-primary-dark);
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid var(--pitch-border);
            border-radius: 10px;
            overflow: hidden;
        }}

        [data-testid="stDataFrame"] th {{
            background: #eef3f8;
            color: var(--pitch-primary-dark);
            font-weight: 600;
        }}

        [data-testid="stSidebarNav"] {{
            padding-top: 0.5rem;
        }}

        [data-testid="stSidebarNav"] span {{
            font-size: 0.94rem;
        }}

        [data-testid="stSidebarNav"] [aria-current="page"] {{
            background: #dce8f7;
            color: var(--pitch-primary-dark);
            border-radius: 7px;
        }}

        .stButton > button {{
            border-color: var(--pitch-primary);
            color: var(--pitch-primary-dark);
            border-radius: 7px;
        }}

        .stButton > button:hover {{
            border-color: var(--pitch-primary-dark);
            color: var(--pitch-primary-dark);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
