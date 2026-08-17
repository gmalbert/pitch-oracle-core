"""1X2 probability header, empirical intervals, and trust badges."""

import math
import streamlit as st

from pitch_oracle_core.ui.formatters import probability


def _value(forecast, name: str, default=None):
    if hasattr(forecast, "get"):
        return forecast.get(name, default)
    return getattr(forecast, name, default)


def render_probability_header(forecast) -> None:
    outcomes = (("Home win", "p_home"), ("Draw", "p_draw"), ("Away win", "p_away"))
    columns = st.columns(3)
    for column, (label, point_name) in zip(columns, outcomes):
        point = float(_value(forecast, point_name, 0.0))
        prefix = point_name.removeprefix("p_")
        lower50 = _value(forecast, f"p_{prefix}_lower50", point)
        upper50 = _value(forecast, f"p_{prefix}_upper50", point)
        lower80 = _value(forecast, f"p_{prefix}_lower80", point)
        upper80 = _value(forecast, f"p_{prefix}_upper80", point)
        column.metric(
            label,
            probability(point),
            help=(
                f"50% model interval: {probability(lower50)}–{probability(upper50)}; "
                f"80% interval: {probability(lower80)}–{probability(upper80)}"
            ),
        )
        column.progress(
            max(0.0, min(1.0, point)),
            text=(
                f"50% {probability(lower50)}–{probability(upper50)} · "
                f"80% {probability(lower80)}–{probability(upper80)}"
            ),
        )
    cold_start = str(_value(forecast, "cold_start", "full"))
    cold_label = str(_value(forecast, "cold_start_label", cold_start.replace("_", " ").title()))
    stability = _value(forecast, "leader_stability", 1.0)
    stability = float(stability) if stability is not None and math.isfinite(float(stability)) else 0.0
    badges = [
        "Full history" if cold_start == "full" else cold_label,
        "Stable" if stability >= 0.75 else "Fragile",
        f"Model {_value(forecast, 'model_id', 'unknown')}",
    ]
    st.caption(" · ".join(badges))
