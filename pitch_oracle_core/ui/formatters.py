"""Presentation-boundary formatting; analytical values stay numeric."""

import math
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import streamlit as st


def probability(value: float | None, digits: int = 0) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}%}"


def signed(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):+.{digits}f}"


def freshness(age_minutes: float) -> tuple[str, str]:
    if age_minutes <= 90:
        return "Fresh", "green"
    if age_minutes <= 360:
        return "Aging", "orange"
    return "Stale", "red"


def timezone_control(competition_timezone: str) -> str:
    """Offer competition-local, explicit user IANA timezone, and UTC display."""
    mode = st.sidebar.selectbox(
        "Kickoff timezone",
        ("Competition local", "My timezone", "UTC"),
        key="pitch_oracle_timezone_mode",
    )
    if mode == "UTC":
        timezone = "UTC"
    elif mode == "Competition local":
        timezone = competition_timezone
    else:
        candidate = st.sidebar.text_input(
            "My IANA timezone", value="America/New_York",
            key="pitch_oracle_user_timezone",
        )
        try:
            ZoneInfo(candidate)
        except ZoneInfoNotFoundError:
            st.sidebar.warning("Unknown timezone; displaying UTC.")
            timezone = "UTC"
        else:
            timezone = candidate
    st.sidebar.caption(f"Kickoffs displayed in {timezone}")
    return timezone
