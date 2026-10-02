"""Shared optional PitchAPI views; no provider calls or model fitting in the app."""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from pitch_oracle_core.pitchapi.artifacts import analytics_repository
from pitch_oracle_core.pitchapi.normalize import boolean
from .context import AppContext


def legacy_context(config, root: str | Path = ".") -> AppContext:
    repository = analytics_repository(root, config.key, os.getenv("PITCH_ORACLE_DATA_DIR", config.data_dir_name))
    return AppContext(config, repository, {}, "PitchAPI", display_timezone=config.sources.weather_timezone)


def latest_revision(frame: pd.DataFrame, groups: list[str] | None = None, clock: str = "observed_at") -> pd.DataFrame:
    """Select a complete response revision, never mix removed rows into it."""
    if frame.empty:
        return frame
    times = pd.to_datetime(frame[clock], utc=True, errors="raise")
    latest = times.groupby([frame[name] for name in groups]).transform("max") if groups else times.max()
    return frame.loc[times.eq(latest)].copy()


def optional_frame(context, name, *, filters=None, latest=True, groups=None, clock="observed_at"):
    if not context.repository.available(name):
        return pd.DataFrame()
    try:
        frame = context.repository.frame(name, filters=filters)
        if latest and "fixture_id" in frame and name not in {"pitchapi_matches", "pitchapi_forecast_revisions", "pitchapi_closing_forecasts"}:
            from pitch_oracle_core.pitchapi.revisions import complete_responses
            revisions = context.repository.frame("pitchapi_response_revisions", filters=filters) if context.repository.available("pitchapi_response_revisions") else pd.DataFrame()
            return complete_responses(frame, revisions, artifact=name, clock=clock)
        return latest_revision(frame, groups, clock) if latest else frame
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.warning(f"{name.removeprefix('pitchapi_').replace('_', ' ').title()} cannot be displayed: {type(exc).__name__}.")
        return pd.DataFrame()


def render_health(context):
    failures = context.repository.manifest.get("optional_artifact_failures", {})
    if failures:
        st.warning("Optional analytics unavailable: " + ", ".join(name.removeprefix("pitchapi_").replace("_", " ") for name in failures))
    if not context.repository.available("pitchapi_health"):
        st.caption("PitchAPI health has not been recorded. Analytics coverage is unknown.")
        return
    health = context.repository.json("pitchapi_health")
    checked = health.get("checked_at", health.get("generated_at", "unknown"))
    st.caption(f"PitchAPI capability audit: {checked}. Historical captures are descriptive unless strict replay is validated.")
    records = [{"Capability": name.replace("_", " ").title(), "Status": item.get("status", "unknown"), "Coverage": item.get("coverage"), "Observed": item.get("observed_at"), "Details": item.get("message", item.get("reason", ""))} for name, item in health.get("capabilities", {}).items()]
    if records:
        st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch", column_config={"Coverage": st.column_config.NumberColumn(format="percent")})


def _fixture_catalog(context):
    matches = optional_frame(context, "pitchapi_matches", latest=False)
    if matches.empty or "fixture_id" not in matches:
        return pd.DataFrame()
    return matches.loc[matches.fixture_id.notna()].sort_values("kickoff_utc", ascending=False).drop_duplicates("fixture_id")


def _provenance(frame, clock="observed_at"):
    if not frame.empty:
        st.caption(f"Observed {pd.to_datetime(frame[clock], utc=True).max().isoformat()} · {len(frame)} rows in this revision.")


def shot_chart(shots, team_labels=None):
    data = shots.dropna(subset=["x", "y"]).copy()
    data["Team"] = data.team_id.map(team_labels or {}).fillna(data.team_id)
    data["Marker xG"] = data.expected_goals.fillna(0)
    data["Outcome"] = data.is_goal.map(lambda value: "Unknown" if pd.isna(value) else "Goal" if boolean(value) else "Shot")
    return alt.Chart(data).mark_point(filled=True, opacity=.8).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0, 105]), title="Pitch length (m) → attacking goal"),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 68]), title="Pitch width (m)"),
        color="Team:N", shape="Outcome:N", size=alt.Size("Marker xG:Q", scale=alt.Scale(domain=[0, 1], range=[25, 650]), title="xG"),
        tooltip=["player_name:N", "minute:Q", "Outcome:N", alt.Tooltip("expected_goals:Q", format=".3f"), alt.Tooltip("expected_goals_on_target:Q", format=".3f")],
    ).properties(height=360)


def _lineups(context, fixture_id):
    frame = optional_frame(context, "pitchapi_lineup_snapshots", filters={"fixture_id": fixture_id}, groups=["team_id"], clock="snapshot_at")
    if frame.empty:
        st.info("No observed lineup is available for this fixture.")
        return
    labels = _team_labels(context)
    for column, (team, members) in zip(st.columns(2), frame.groupby("team_id", sort=False)):
        with column:
            status = str(members.iloc[0].lineup_status)
            st.subheader(f"{labels.get(team, team)} · {status}")
            _provenance(members, "snapshot_at")
            st.caption(f"Formation: {members.iloc[0].get('formation') or 'unknown'}")
            fields = [name for name in ("player_name", "position", "is_starter") if name in members]
            st.dataframe(members.sort_values("is_starter", ascending=False)[fields], hide_index=True, width="stretch")
            st.caption("Position categories may be unknown; formation slots do not identify player roles.")


def _forecast_history(context, fixture_id):
    revisions = optional_frame(context, "pitchapi_forecast_revisions", filters={"fixture_id": fixture_id}, latest=False)
    if revisions.empty:
        st.info("No archived pre-match forecast issues are available.")
        return
    revisions = revisions.sort_values("issued_at")
    st.dataframe(revisions[[name for name in ("issued_at", "revision_label", "p_home", "p_draw", "p_away", "home_lineup_status", "away_lineup_status", "model_id", "enabled_families", "fallback_reason") if name in revisions]], hide_index=True, width="stretch")
    st.line_chart(revisions.assign(issued_at=pd.to_datetime(revisions.issued_at, utc=True)), x="issued_at", y=["p_home", "p_draw", "p_away"])
    closing = optional_frame(context, "pitchapi_closing_forecasts", filters={"fixture_id": fixture_id}, latest=False)
    if not closing.empty:
        st.caption(f"Closing forecast: actual issue {closing.iloc[-1].issued_at}; selected after kickoff without new inference.")
    latest = revisions.iloc[-1]
    context_rows = []
    for side in ("home", "away"):
        context_rows.append({"Team": side.title(), **{name.replace("_", " "): latest.get(f"{side}_{name}") for name in ("lineup_attack_delta", "lineup_defense_delta", "lineup_coverage", "lineup_continuity", "lineup_missing_starter_value", "keeper_strength", "keeper_effective_minutes", "keeper_shots_on_target_faced")}})
    st.caption("Lineup and goalkeeper context recorded at the latest forecast issue. Unknown values stay blank.")
    st.dataframe(pd.DataFrame(context_rows), hide_index=True, width="stretch")


def _network(context, fixture_id):
    frame = optional_frame(context, "pitchapi_network", filters={"fixture_id": fixture_id})
    if frame.empty:
        st.info("Passing network unavailable.")
        return
    labels = _team_labels(context)
    selected = st.selectbox("Network team", sorted(frame.team_id.unique()), format_func=lambda team: labels.get(team, team), key="pitchapi_network_team")
    frame = frame.loc[frame.team_id == selected]
    nodes = frame.loc[frame.kind == "node"].dropna(subset=["player_id", "avg_x", "avg_y"])
    edges = frame.loc[frame.kind == "edge"].reindex(columns=["from_player_id", "to_player_id", "passes"]).merge(nodes[["player_id", "avg_x", "avg_y"]].rename(columns={"player_id": "from_player_id", "avg_x": "x", "avg_y": "y"}), on="from_player_id").merge(nodes[["player_id", "avg_x", "avg_y"]].rename(columns={"player_id": "to_player_id", "avg_x": "x2", "avg_y": "y2"}), on="to_player_id")
    lines = alt.Chart(edges).mark_rule(opacity=.45).encode(x=alt.X("x:Q", scale=alt.Scale(domain=[0, 105]), title="Pitch length (m) → attacking goal"), y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 68]), title="Pitch width (m)"), x2="x2:Q", y2="y2:Q", strokeWidth=alt.StrokeWidth("passes:Q", title="Passes"), tooltip=["from_player_id:N", "to_player_id:N", "passes:Q"])
    points = alt.Chart(nodes).mark_circle(size=180).encode(x=alt.X("avg_x:Q", title="Pitch length (m) → attacking goal"), y=alt.Y("avg_y:Q", title="Pitch width (m)"), tooltip=["player_name:N", "passes:Q", "passes_received:Q"])
    st.altair_chart((lines + points).properties(height=360), width="stretch")
    _provenance(frame)
    st.caption("Both teams attack left to right in their own coordinate frame.")


def _heatmaps(context, fixture_id):
    frame = optional_frame(context, "pitchapi_heatmaps", filters={"fixture_id": fixture_id})
    if frame.empty:
        st.info("Heatmaps unavailable.")
        return
    members = frame[["team_id", "player_id", "player_name", "kind"]].drop_duplicates().reset_index(drop=True)
    teams = _team_labels(context)
    labels = members.apply(lambda row: f"{teams.get(row.team_id, row.team_id)} · {row.player_name if pd.notna(row.player_name) else 'Team'}", axis=1)
    index = st.selectbox("Heatmap subject", range(len(members)), format_func=lambda i: labels.iloc[i])
    member = members.iloc[index]
    selected = frame.loc[frame.team_id.eq(member.team_id) & (frame.player_id.eq(member.player_id) if pd.notna(member.player_id) else frame.player_id.isna())]
    chart = alt.Chart(selected).mark_rect().encode(x=alt.X("cell_x:O", title="Pitch length → attacking goal"), y=alt.Y("cell_y:O", title="Pitch width"), color=alt.Color("actions:Q", title="Actions"), tooltip=["cell_x:O", "cell_y:O", "actions:Q"])
    st.altair_chart(chart.properties(height=340), width="stretch")
    _provenance(selected)
    st.caption(f"Coordinate frame: {selected.iloc[0].coordinate_frame}; grid {selected.iloc[0].grid_length} × {selected.iloc[0].grid_width}.")


def render_match_detail(context, fixture_id):
    section = st.segmented_control("Match detail", ["Lineups", "Forecast history", "Shots", "Momentum", "Team comparison", "Players", "Passing network", "Heatmaps"], default="Lineups", key="pitchapi_match_detail") or "Lineups"
    st.subheader(section)
    if section == "Lineups":
        _lineups(context, fixture_id)
    elif section == "Forecast history":
        _forecast_history(context, fixture_id)
    elif section == "Shots":
        summary = optional_frame(context, "pitchapi_match_shot_features", filters={"fixture_id": fixture_id})
        if not summary.empty:
            metrics = ["xg", "xgot", "shots", "xg_per_shot", "big_chances", "chance_concentration"]
            st.dataframe(pd.DataFrame([{ "Team": side.title(), **{metric: summary.iloc[0].get(f"{side}_{metric}") for metric in metrics}} for side in ("home", "away")]), hide_index=True, width="stretch")
        shots = optional_frame(context, "pitchapi_shots", filters={"fixture_id": fixture_id})
        if shots.empty:
            st.info("Shot events unavailable. A known zero-shot result is shown only when a complete response was observed.")
        else:
            st.altair_chart(shot_chart(shots, _team_labels(context)), width="stretch")
            _provenance(shots)
            st.caption("Provider coordinates: 105 × 68 metres; each team attacks left to right. Marker size shows xG; missing xG is unknown.")
    elif section == "Momentum":
        frame = optional_frame(context, "pitchapi_momentum", filters={"fixture_id": fixture_id})
        series = [name for name in ("home", "away", "home_value", "away_value", "value") if name in frame]
        if frame.empty or not series:
            st.info("Momentum curve unavailable.")
        else:
            st.line_chart(frame.sort_values("minute"), x="minute", y=series)
            _provenance(frame)
    elif section in {"Team comparison", "Players"}:
        artifact = "pitchapi_advanced_team" if section == "Team comparison" else "pitchapi_player_match"
        frame = optional_frame(context, artifact, filters={"fixture_id": fixture_id})
        if frame.empty:
            st.info(f"{section} unavailable.")
        else:
            fields = [name for name in ("team_id", "player_name", "position", "minutes", "rating", "xt_total", "vaep_offensive", "vaep_defensive", "passes", "pass_accuracy", "progressive_passes", "ppda", "field_tilt", "direct_speed", "xag") if name in frame and frame[name].notna().any()]
            if "team_id" in frame:
                frame["team_id"] = frame.team_id.map(_team_labels(context)).fillna(frame.team_id)
            st.dataframe(frame[fields], hide_index=True, width="stretch")
            _provenance(frame)
    elif section == "Passing network":
        _network(context, fixture_id)
    elif section == "Heatmaps":
        _heatmaps(context, fixture_id)


def render_match_page(context):
    st.title("Match analytics")
    render_health(context)
    matches = _fixture_catalog(context)
    if matches.empty:
        st.info("No reconciled PitchAPI fixtures are available yet. Existing predictions remain available.")
        return
    options = matches.fixture_id.astype(str).tolist()
    labels = {str(row.fixture_id): f"{row.home_team} vs {row.away_team} · {pd.Timestamp(row.kickoff_utc).tz_convert(context.display_timezone).strftime('%b %d, %H:%M %Z')} · {row.status}" for row in matches.itertuples()}
    requested = str(st.query_params.get("fixture", ""))
    selected = st.selectbox("Analytics fixture", options, index=options.index(requested) if requested in options else 0, format_func=labels.get)
    st.query_params["fixture"] = selected
    render_match_detail(context, selected)


STYLE_METRICS = ("ppda", "field_tilt", "direct_speed", "progressive_passes", "xt_total", "vaep_offensive", "vaep_defensive")


def _team_labels(context):
    matches = _fixture_catalog(context)
    return dict(zip(matches.home_team_id, matches.home_team)) | dict(zip(matches.away_team_id, matches.away_team)) if not matches.empty else {}


def style_percentiles(frame):
    available = [name for name in STYLE_METRICS if name in frame and frame[name].notna().any()]
    means = frame.groupby("team_id")[available].mean()
    return means.rank(pct=True).assign(**({"ppda": means.ppda.rank(pct=True, ascending=False)} if "ppda" in means else {}))


def render_team_page(context):
    st.title("Team analytics")
    render_health(context)
    matches = _fixture_catalog(context)
    if matches.empty:
        st.info("Team analytics will appear after reconciled observations are collected.")
        return
    labels = dict(zip(matches.home_team_id, matches.home_team)) | dict(zip(matches.away_team_id, matches.away_team))
    teams = sorted(labels, key=lambda key: labels[key])
    team = st.selectbox("Analytics team", teams, format_func=labels.get)
    section = st.segmented_control("Team detail", ["Trends", "Style radar", "Players"], default="Trends")
    if section == "Players":
        frame = optional_frame(context, "pitchapi_player_match", filters={"team_id": team}, groups=["fixture_id"])
        if frame.empty:
            st.info("Observed player history unavailable.")
            return
        recent = frame.sort_values("observed_at").groupby("player_id").agg(player_name=("player_name", "last"), minutes=("minutes", "sum"), matches=("fixture_id", "nunique"))
        st.caption("Observed minutes and appearances; these totals do not estimate player ability.")
        st.dataframe(recent.sort_values("minutes", ascending=False), width="stretch")
        return
    advanced = optional_frame(context, "pitchapi_advanced_team", filters={"team_id": team} if section == "Trends" else None, groups=["fixture_id", "team_id"])
    if advanced.empty:
        st.info("Advanced team history unavailable.")
        return
    advanced = advanced.merge(matches[["fixture_id", "kickoff_utc"]], on="fixture_id", how="inner", validate="many_to_one").sort_values("kickoff_utc").groupby("team_id", group_keys=False).tail(10)
    if section == "Style radar":
        ranks = style_percentiles(advanced)
        if team not in ranks.index or len(ranks.columns) < 3:
            st.info("At least three observed style metrics are needed for a radar.")
            return
        values = ranks.loc[team].dropna()
        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
        data = pd.DataFrame({"Metric": values.index, "Percentile": values.values, "x": values.values * np.cos(angles), "y": values.values * np.sin(angles), "order": range(len(values))})
        data = pd.concat([data, data.iloc[[0]].assign(order=len(values))], ignore_index=True)
        chart = alt.Chart(data).mark_line(point=True).encode(x=alt.X("x:Q", scale=alt.Scale(domain=[-1.1, 1.1]), axis=None), y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.1, 1.1]), axis=None), order="order:Q", tooltip=["Metric:N", alt.Tooltip("Percentile:Q", format=".0%")])
        axes = pd.DataFrame({"Metric": values.index, "x": np.cos(angles), "y": np.sin(angles), "origin": 0})
        spokes = alt.Chart(axes).mark_rule(opacity=.3).encode(x="origin:Q", y="origin:Q", x2="x:Q", y2="y:Q")
        text = alt.Chart(axes).mark_text(dy=-10).encode(x="x:Q", y="y:Q", text="Metric:N")
        st.altair_chart((spokes + chart + text).properties(height=400), width="stretch")
        st.caption(f"Percentiles among {len(ranks)} observed teams, using each team's latest 10 covered matches. Lower PPDA ranks higher. Coverage can differ between teams.")
        st.dataframe(values.rename("Percentile"), width="stretch")
    else:
        candidates = [name for name in STYLE_METRICS + ("sequence_time", "buildup_attacks", "direct_attacks", "sca", "xag") if name in advanced and advanced[name].notna().any()]
        if candidates:
            metric = st.selectbox("Team trend metric", candidates, format_func=lambda name: name.replace("_", " ").title())
            st.line_chart(advanced.assign(kickoff_utc=pd.to_datetime(advanced.kickoff_utc, utc=True)), x="kickoff_utc", y=metric)
            st.caption(f"{len(advanced)} covered matches; missing observations are excluded, never filled with zero.")
        summaries = optional_frame(context, "pitchapi_match_shot_features", filters={"fixture_id": matches.loc[matches.home_team_id.eq(team) | matches.away_team_id.eq(team), "fixture_id"].tolist()}, groups=["fixture_id"])
        if not summaries.empty:
            summaries = summaries.merge(matches[["fixture_id", "kickoff_utc", "home_team_id"]], on="fixture_id", validate="one_to_one").sort_values("kickoff_utc").tail(10)
            for metric in ("xg_per_shot", "chance_concentration", "finishing_vs_expectation"):
                summaries[metric] = np.where(summaries.home_team_id.eq(team), summaries.get(f"home_{metric}", np.nan), summaries.get(f"away_{metric}", np.nan))
            st.subheader("Chance quality")
            st.line_chart(summaries.assign(kickoff_utc=pd.to_datetime(summaries.kickoff_utc, utc=True)), x="kickoff_utc", y=["xg_per_shot", "chance_concentration", "finishing_vs_expectation"])


def render_model_page(context):
    from pitch_oracle_core.features.families import FAMILIES
    st.title("Feature validation")
    render_health(context)
    configuration = context.repository.manifest.get("pitchapi", {})
    enabled = set(configuration.get("enabled_families", []))
    st.dataframe(pd.DataFrame([{"Family": family.replace("_", " ").title(), "In production": family in enabled, "Evidence": configuration.get("evidence_id") if family in enabled else None, "Reason": "Passed league promotion gates" if family in enabled else "Baseline retained until strict walk-forward gates pass"} for family in FAMILIES]), hide_index=True, width="stretch")
    if context.repository.available("pitchapi_ablation"):
        report = context.repository.json("pitchapi_ablation")
        st.json(report)
    else:
        st.info("No PitchAPI feature ablation report has been published for this league.")
