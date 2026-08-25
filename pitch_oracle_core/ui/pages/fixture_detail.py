"""Streamlit page: Fixture Detail — every market for the selected fixture."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from pitch_oracle_core.markets.grid import market_grid_from_lambdas


def render(context):
    """Render the Fixture Detail page."""
    st.title("Fixture Detail")
    st.caption("Full market surface for a selected fixture")

    data_dir = Path(context.config.data_dir_name)
    preds_path = data_dir / "upcoming_predictions.csv"

    if not preds_path.exists():
        st.info("No upcoming predictions available.")
        return

    preds = pd.read_csv(preds_path)
    if preds.empty:
        st.info("No upcoming fixtures.")
        return

    # Fixture selector
    fixtures = preds.apply(
        lambda r: f"{r.get('HomeTeam', '')} vs {r.get('AwayTeam', '')}", axis=1
    ).tolist()
    selected = st.selectbox("Select fixture", fixtures)

    if not selected:
        return

    idx = fixtures.index(selected)
    row = preds.iloc[idx]

    st.subheader(selected)

    # Show model probabilities
    home_prob = row.get("HomeWin_Prob", row.get("PredHomeWin", 0))
    draw_prob = row.get("Draw_Prob", row.get("PredDraw", 0))
    away_prob = row.get("AwayWin_Prob", row.get("PredAwayWin", 0))

    col1, col2, col3 = st.columns(3)
    col1.metric("Home Win", f"{float(home_prob):.1%}")
    col2.metric("Draw", f"{float(draw_prob):.1%}")
    col3.metric("Away Win", f"{float(away_prob):.1%}")

    # Goal markets
    st.subheader("Goal Markets")
    exp_home = row.get("ExpHomeGoals", 1.4)
    exp_away = row.get("ExpAwayGoals", 1.1)
    try:
        grid = market_grid_from_lambdas(float(exp_home), float(exp_away))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Over 2.5", f"{grid['over_2_5']:.1%}")
        col2.metric("Under 2.5", f"{grid['under_2_5']:.1%}")
        col3.metric("BTTS Yes", f"{grid['btts_yes']:.1%}")
        col4.metric("BTTS No", f"{grid['btts_no']:.1%}")

        col1, col2 = st.columns(2)
        col1.metric("Expected Home Goals", f"{grid['exp_home_goals']:.2f}")
        col2.metric("Expected Away Goals", f"{grid['exp_away_goals']:.2f}")
    except Exception:
        st.warning("Could not compute goal markets")

    # Risk assessment
    risk_score = row.get("Risk_Score", None)
    if risk_score is not None:
        st.subheader("Risk Assessment")
        st.metric("Risk Score", f"{float(risk_score):.0f}/100")
        st.caption(row.get("Recommendation", ""))
