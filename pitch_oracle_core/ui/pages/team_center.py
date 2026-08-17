"""Team command center: form, splits, strength, schedule, and optional context."""

import pandas as pd
import streamlit as st

from pitch_oracle_core.ui.components.team_trends import render_team_trend


def render(context) -> None:
    st.title("Team Command Center")
    snapshots = context.repository.frame("team_snapshots")
    label_column = "team_name" if "team_name" in snapshots else "team_id"
    labels = snapshots[label_column].astype(str).tolist()
    requested = st.query_params.get("team")
    requested = requested[0] if isinstance(requested, list) and requested else requested
    ids = snapshots.team_id.astype(str).tolist()
    default_index = ids.index(str(requested)) if str(requested) in ids else 0
    selected_label = st.selectbox("Team", labels, index=default_index)
    team = snapshots.loc[snapshots[label_column].astype(str) == selected_label].iloc[0]
    team_id = str(team.team_id)
    st.query_params["team"] = team_id
    columns = st.columns(4)
    columns[0].metric("Power rank", int(team.get("power_rank", team.get("current_position", 0))) or "—")
    columns[1].metric("Elo", f"{float(team.get('elo_rating', 1500)):.0f}")
    columns[2].metric("Last 5 points", f"{float(team.get('points_l5', 0)):.0f}")
    columns[3].metric("Matches", int(team.get("matches", 0)))
    if pd.notna(team.get("projection_expected_position")):
        st.caption(
            f"Season projection snapshot: expected rank "
            f"{float(team.get('projection_expected_position')):.1f}, "
            f"expected points {float(team.get('projection_expected_points')):.1f}."
        )
    st.subheader("Form fingerprint")
    fingerprint_fields = (
        ("Attack", "attack_l10"), ("Defense", "defense_l10"),
        ("Goal difference", "goal_difference_l10"),
        ("Clean sheets", "clean_sheet_rate_l10"),
        ("xG / shot proxy", "xg_for_ewm10"),
        ("Shots", "shots_for_ewm10"),
        ("Shot quality", "shot_quality_ewm10"),
        ("Finishing vs expectation", "finishing_vs_expectation_ewm10"),
        ("Opponent-adjusted points", "opponent_adjusted_points_l10"),
    )
    fingerprint = pd.DataFrame([
        {"Metric": label, "Value": team.get(field)}
        for label, field in fingerprint_fields if pd.notna(team.get(field))
    ])
    st.dataframe(fingerprint, hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.metric(
        f"Home attack · n={int(team.get('home_sample', 0))}",
        f"{float(team.get('home_goals_for_shrunk', 0)):.2f}",
    )
    b.metric(
        f"Away attack · n={int(team.get('away_sample', 0))}",
        f"{float(team.get('away_goals_for_shrunk', 0)):.2f}",
    )
    if context.repository.available("team_events"):
        events = context.repository.frame("team_events")
        events = events.loc[events.team_id.astype(str) == team_id]
        window = st.segmented_control("Window", [5, 10, 20, "All"], default=10)
        role = st.segmented_control("Venue", ["overall", "home", "away"], default="overall")
        if role != "overall":
            events = events.loc[events.venue_role == role]
        if window != "All":
            events = events.sort_values("kickoff_utc").tail(int(window))
        metric_options = [
            item for item in ("points", "goals_for", "goals_against", "opponent_adjusted_points")
            if item in events.columns
        ]
        if metric_options:
            metric = st.selectbox("Trend metric", metric_options)
            render_team_trend(events, selected_label, metric, metric.replace("_", " ").title())
        st.caption(f"n={len(events)} completed team-perspective matches")
        recent_columns = [
            column for column in (
                "kickoff_utc", "venue_role", "opponent_name", "score", "result",
                "xg_for", "shots_for", "shot_quality", "finishing_vs_expectation",
            ) if column in events
        ]
        if recent_columns:
            st.subheader("Recent matches")
            st.dataframe(
                events.sort_values("kickoff_utc", ascending=False)[recent_columns].head(10),
                hide_index=True,
                width="stretch",
            )
    for artifact, title in (
        ("fixture_difficulty", "Fixture difficulty calendar"),
        ("style_fingerprints", "Style fingerprint"),
        ("recovery_load", "Congestion, travel, and recovery load"),
        ("squad_availability", "Squad availability"),
        ("manager_effects", "Manager change tracker"),
        ("referee_matchups", "Discipline and referee matchup"),
    ):
        if context.repository.available(artifact):
            frame = context.repository.frame(artifact)
            if "team_id" in frame:
                frame = frame.loc[frame.team_id.astype(str) == team_id]
            with st.expander(title):
                st.dataframe(frame, hide_index=True, width="stretch")
