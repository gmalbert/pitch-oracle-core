"""Shared visual theme for Pitch Oracle Streamlit consumers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import streamlit as st

from .config import LeagueConfig

# Shared palette choices exposed by the consumer sidebar. The browser-local
# clock supplies the initial daytime/nighttime selection; users can override it.
LAUNCH_THEMES: dict[str, dict[str, str]] = {
    # ── Daytime · light themes ────────────────────────────────────────────
    "☀️ Daytime · Alpine Mist": {
        "primary": "#3f6b8e",
        "primary_dark": "#26424f",
        "sidebar": "#eef3f6",
        "page": "#fbfdfe",
        "border": "#d3e0e8",
        "muted": "#5f7484",
    },
    "☀️ Daytime · Amsterdam Canal": {
        "primary": "#156b5b",
        "primary_dark": "#0b3d33",
        "sidebar": "#eaf4f1",
        "page": "#f7fbfa",
        "border": "#cfe5de",
        "muted": "#5f7a72",
    },
    "☀️ Daytime · Apricot Studio": {
        "primary": "#c96f4a",
        "primary_dark": "#7e3f26",
        "sidebar": "#faf1ea",
        "page": "#fefaf7",
        "border": "#f0d8c8",
        "muted": "#826c5f",
    },
    "☀️ Daytime · Blue Sky Ledger": {
        "primary": "#2a74d0",
        "primary_dark": "#14477e",
        "sidebar": "#edf4fc",
        "page": "#fbfdff",
        "border": "#d2e3f5",
        "muted": "#5f7a99",
    },
    "☀️ Daytime · Botanical Field": {
        "primary": "#4c8a3f",
        "primary_dark": "#2a5222",
        "sidebar": "#eff6eb",
        "page": "#fbfdf9",
        "border": "#d7e8cf",
        "muted": "#647c5c",
    },
    "☀️ Daytime · Citrus Press": {
        "primary": "#d9822b",
        "primary_dark": "#8a4d12",
        "sidebar": "#faf3e8",
        "page": "#fefbf5",
        "border": "#f0dfc6",
        "muted": "#86745c",
    },
    "☀️ Daytime · Cloudline": {
        "primary": "#7a8699",
        "primary_dark": "#46505f",
        "sidebar": "#f1f4f8",
        "page": "#fdfefe",
        "border": "#dbe1ea",
        "muted": "#6b7686",
    },
    "☀️ Daytime · Delft Blue": {
        "primary": "#1e5aa8",
        "primary_dark": "#103a6b",
        "sidebar": "#eaf1fa",
        "page": "#f8fbfe",
        "border": "#cfdff3",
        "muted": "#5c7391",
    },
    "☀️ Daytime · Desert Paper": {
        "primary": "#b07840",
        "primary_dark": "#714822",
        "sidebar": "#f7f1e7",
        "page": "#fdfbf7",
        "border": "#eadbc4",
        "muted": "#83725b",
    },
    "☀️ Daytime · Glacier Glass": {
        "primary": "#4a9aa8",
        "primary_dark": "#285d66",
        "sidebar": "#edf7f8",
        "page": "#fafdfd",
        "border": "#d2e9ed",
        "muted": "#5d7b80",
    },
    "☀️ Daytime · Linen & Ink": {
        "primary": "#33415c",
        "primary_dark": "#1c2536",
        "sidebar": "#f3f1ec",
        "page": "#fdfcf9",
        "border": "#e0dcd2",
        "muted": "#6f6a60",
    },
    "☀️ Daytime · Meadow Scoreboard": {
        "primary": "#2f7d4f",
        "primary_dark": "#19492c",
        "sidebar": "#ecf5ef",
        "page": "#f9fcf9",
        "border": "#d3e8db",
        "muted": "#5e7a68",
    },
    "☀️ Daytime · Nordic Slate": {
        "primary": "#5b6b7d",
        "primary_dark": "#333f4c",
        "sidebar": "#eef1f4",
        "page": "#fbfcfd",
        "border": "#d8dee6",
        "muted": "#68737f",
    },
    "☀️ Daytime · Sandstone Matchday": {
        "primary": "#b0713f",
        "primary_dark": "#6f4220",
        "sidebar": "#f8f1e9",
        "page": "#fdfbf8",
        "border": "#ecdbca",
        "muted": "#84705c",
    },
    "☀️ Daytime · Tulip Terrace": {
        "primary": "#c0507a",
        "primary_dark": "#7c2c4d",
        "sidebar": "#fbeef3",
        "page": "#fefafc",
        "border": "#f2d3e0",
        "muted": "#85616f",
    },
    # ── Nighttime · dark themes ────────────────────────────────────────────
    "🌙 Nighttime · Aurora Floodlights": {
        "primary": "#5fd0a8",
        "primary_dark": "#a8f0d5",
        "sidebar": "#0d1f1a",
        "page": "#081410",
        "border": "#1d3a31",
        "muted": "#8fb8a8",
    },
    "🌙 Nighttime · Blackout Pitch": {
        "primary": "#7f8c9b",
        "primary_dark": "#c8d0da",
        "sidebar": "#11151a",
        "page": "#0a0c10",
        "border": "#242b34",
        "muted": "#8d97a3",
    },
    "🌙 Nighttime · Blue Hour": {
        "primary": "#6b9fe0",
        "primary_dark": "#b8d4f5",
        "sidebar": "#101f38",
        "page": "#0b1628",
        "border": "#22375a",
        "muted": "#8ea7c8",
    },
    "🌙 Nighttime · Carbon & Lime": {
        "primary": "#a4c43c",
        "primary_dark": "#d3e88a",
        "sidebar": "#15180e",
        "page": "#0e100a",
        "border": "#2a2f1c",
        "muted": "#9aa57c",
    },
    "🌙 Nighttime · City Neon": {
        "primary": "#e066d6",
        "primary_dark": "#f2a8ec",
        "sidebar": "#241026",
        "page": "#180a1a",
        "border": "#402046",
        "muted": "#b18bb5",
    },
    "🌙 Nighttime · Deep Sea": {
        "primary": "#3f9ad9",
        "primary_dark": "#93cdf0",
        "sidebar": "#0c2031",
        "page": "#081722",
        "border": "#1b3a54",
        "muted": "#7fa3bd",
    },
    "🌙 Nighttime · Midnight Oranje": {
        "primary": "#ff7a1a",
        "primary_dark": "#ffb066",
        "sidebar": "#231005",
        "page": "#170a03",
        "border": "#3f230f",
        "muted": "#c09b80",
    },
    "🌙 Nighttime · Moonlit Turf": {
        "primary": "#57b46f",
        "primary_dark": "#a2ddb2",
        "sidebar": "#0e1f13",
        "page": "#09150c",
        "border": "#1e3a27",
        "muted": "#8fb29b",
    },
    "🌙 Nighttime · Night Watch": {
        "primary": "#c9a227",
        "primary_dark": "#e8cd74",
        "sidebar": "#1e1a08",
        "page": "#141105",
        "border": "#383316",
        "muted": "#b0a47c",
    },
    "🌙 Nighttime · Obsidian Gold": {
        "primary": "#d9a441",
        "primary_dark": "#edc97e",
        "sidebar": "#191408",
        "page": "#100d05",
        "border": "#332a12",
        "muted": "#ab9c77",
    },
    "🌙 Nighttime · Purple Rain": {
        "primary": "#9b6ee0",
        "primary_dark": "#c7a8f0",
        "sidebar": "#1a1129",
        "page": "#110b1c",
        "border": "#312149",
        "muted": "#a08fb8",
    },
    "🌙 Nighttime · Stadium Shadow": {
        "primary": "#8a94a6",
        "primary_dark": "#c3cad6",
        "sidebar": "#14171c",
        "page": "#0d0f13",
        "border": "#272c35",
        "muted": "#8e97a3",
    },
    "🌙 Nighttime · Velvet Navy": {
        "primary": "#5b7bd4",
        "primary_dark": "#a3b8ea",
        "sidebar": "#0e1426",
        "page": "#090d1a",
        "border": "#1e2a4a",
        "muted": "#8793b8",
    },
    "🌙 Nighttime · Voltage Violet": {
        "primary": "#b26be0",
        "primary_dark": "#d5a8f0",
        "sidebar": "#1b1026",
        "page": "#120a1a",
        "border": "#322145",
        "muted": "#a18ab8",
    },
    "🌙 Nighttime · Winter Night": {
        "primary": "#7fb3e6",
        "primary_dark": "#bcd8f5",
        "sidebar": "#0e1c2e",
        "page": "#091420",
        "border": "#1e344e",
        "muted": "#8aa3be",
    },
}

DAY_THEME_NAME = "☀️ Daytime · Delft Blue"
NIGHT_THEME_NAME = "🌙 Nighttime · Winter Night"
DAY_START_HOUR = 7
NIGHT_START_HOUR = 19

def _theme_name_for_hour(local_hour: int) -> str:
    """Select the fixed production palette for a browser-local hour."""
    if not 0 <= local_hour <= 23:
        raise ValueError("local_hour must be between 0 and 23")
    return DAY_THEME_NAME if DAY_START_HOUR <= local_hour < NIGHT_START_HOUR else NIGHT_THEME_NAME


def _browser_local_hour(
    *,
    now_utc: datetime | None = None,
    timezone_name: str | None = None,
    timezone_offset_minutes: int | None = None,
) -> int:
    """Return the current hour in the browser's timezone with safe fallbacks."""
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if timezone_name is None:
        timezone_name = st.context.timezone
    if timezone_offset_minutes is None:
        timezone_offset_minutes = st.context.timezone_offset
    if timezone_name:
        try:
            return now.astimezone(ZoneInfo(timezone_name)).hour
        except (ZoneInfoNotFoundError, ValueError):
            pass
    if timezone_offset_minutes is not None:
        browser_timezone = timezone(-timedelta(minutes=timezone_offset_minutes))
        return now.astimezone(browser_timezone).hour
    return now.astimezone().hour


def apply_theme(config: LeagueConfig) -> None:
    """Apply the shared Pitch Oracle visual system.

    Consumers provide the league identity; the core owns the common visual
    language so every league deployment feels like the same product.

    The browser-local clock selects Delft Blue from 07:00 through 18:59 and
    Winter Night overnight.
    """
    choice = _theme_name_for_hour(_browser_local_hour())
    palette = LAUNCH_THEMES[choice]
    primary = palette["primary"]
    primary_dark = palette["primary_dark"]
    sidebar = palette["sidebar"]
    page = palette["page"]
    border = palette["border"]
    muted = palette["muted"]

    # Dark themes need light text everywhere; infer from the page background
    # luminance so the palettes can't drift out of sync with the CSS mode.
    def _luminance(hex_color: str) -> float:
        value = hex_color.lstrip("#")
        try:
            r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return 1.0
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

    dark = _luminance(page) < 0.5

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
            --pitch-text: {'#e8ecf2' if dark else '#31333f'};
            --pitch-text-soft: {'#aab4c2' if dark else '#5b6472'};
            --pitch-card: {'#14181f' if dark else '#f8fafc'};
            --pitch-card-border: {'#2a313c' if dark else '#e5eaf0'};
            --pitch-header-bg: {'#1a2029' if dark else '#eef3f8'};
            --pitch-active-bg: {'#243040' if dark else '#dce8f7'};
        }}

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {{
            background: var(--pitch-page);
            color: var(--pitch-text);
        }}

        /* Keep Streamlit's fixed top bar on the same theme surface as the
           main section.  The explicit toolbar/decorative selectors prevent
           Streamlit's default secondary-background color from showing
           through in either light or dark mode. */
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {{
            background: var(--pitch-page) !important;
            background-color: var(--pitch-page) !important;
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

        /* Streamlit's own heading rule (`.st-emotion-cache-* h1`) beats a bare
           `h1`, so scope headings to the app container to win the cascade. */
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stAppViewContainer"] h5,
        [data-testid="stAppViewContainer"] h6 {{
            color: var(--pitch-primary-dark);
            letter-spacing: -0.02em;
        }}

        [data-testid="stAppViewContainer"] h1 {{
            margin-bottom: 0.35rem;
        }}

        [data-testid="stCaptionContainer"] {{
            color: var(--pitch-muted);
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {{
            color: var(--pitch-text);
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
            background: var(--pitch-card);
            border: 1px solid var(--pitch-card-border);
            border-radius: 10px;
            padding: 0.85rem 1rem;
        }}

        [data-testid="stMetricLabel"] {{
            color: var(--pitch-text-soft);
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
            background: var(--pitch-header-bg);
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
            background: var(--pitch-active-bg);
            color: var(--pitch-primary-dark);
            border-radius: 7px;
        }}

        .stButton > button {{
            border-color: var(--pitch-primary);
            color: var(--pitch-primary-dark);
            background: transparent;
            border-radius: 7px;
        }}

        .stButton > button:hover {{
            border-color: var(--pitch-primary-dark);
            color: var(--pitch-primary-dark);
        }}

        [data-testid="stDownloadButton"] > button {{
            border-color: var(--pitch-primary);
            color: var(--pitch-primary-dark);
            background: transparent;
            border-radius: 7px;
        }}

        [data-testid="stDownloadButton"] > button:hover {{
            border-color: var(--pitch-primary-dark);
            color: var(--pitch-primary-dark);
        }}

        [data-testid="stDownloadButton"] > button * {{
            color: inherit;
        }}

        /* ── Mode-aware component text (vars flip between light and dark) ── */
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"] {{
            color: var(--pitch-text);
        }}

        [data-testid="stSidebarNav"] a,
        [data-testid="stSidebarNav"] span {{
            color: var(--pitch-text);
        }}

        [data-testid="stDataFrame"] .glide-cell {{
            color: var(--pitch-text);
        }}

        .stTabs [data-baseweb="tab-list"] button {{
            color: var(--pitch-text-soft);
        }}

        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            color: var(--pitch-primary-dark);
        }}

        /* st.segmented_control renders as a button group rather than .stTabs. */
        [data-testid="stButtonGroup"] [data-variant="segmented_control"] {{
            color: var(--pitch-text-soft);
            background: var(--pitch-card);
            border-color: var(--pitch-card-border);
        }}

        [data-testid="stButtonGroup"] [data-variant="segmented_control"] * {{
            color: inherit;
        }}

        [data-testid="stButtonGroup"] [data-variant="segmented_control"][aria-checked="true"] {{
            color: var(--pitch-primary-dark);
            background: var(--pitch-active-bg);
        }}

        label,
        .stRadio label,
        .stCheckbox label,
        .stSelectbox label {{
            color: var(--pitch-text);
        }}

        input,
        textarea,
        [data-baseweb="select"] > div {{
            color: var(--pitch-text);
            background-color: var(--pitch-card);
            border-color: var(--pitch-card-border);
        }}

        [data-baseweb="select"] * {{
            color: var(--pitch-text);
        }}

        .stAlert {{
            color: var(--pitch-text);
        }}

        /* Expander header text must follow the theme in both open and closed
           states.  Streamlit's default can leave the summary white in dark
           mode, which is unreadable on light themes. */
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span {{
            color: var(--pitch-text) !important;
        }}

        [data-testid="stExpander"] details[open] summary {{
            color: var(--pitch-text) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
