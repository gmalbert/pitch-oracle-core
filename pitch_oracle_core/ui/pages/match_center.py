"""Fixture intelligence center and scenario sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
import html
from urllib.parse import quote
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st

from pitch_oracle_core.domain.forecasts import markets_from_score_matrix
from pitch_oracle_core.ui.components.drivers import evidence_bullets, render_drivers
from pitch_oracle_core.ui.components.freshness import render_capability
from pitch_oracle_core.ui.components.probability import render_probability_header
from pitch_oracle_core.ui.components.score_matrix import goal_market_frame, render_score_matrix
from pitch_oracle_core.ui.scenarios import ScenarioControl


def selected_fixture(fixtures: pd.DataFrame, display_timezone: str = "UTC") -> str:
    if fixtures.empty:
        raise ValueError("No fixtures are available")
    requested = st.query_params.get("fixture")
    valid_ids = set(fixtures.fixture_id.astype(str))
    ids = fixtures.fixture_id.astype(str).tolist()
    default_index = ids.index(str(requested)) if requested in valid_ids else 0
    labels = fixtures.apply(
        lambda row: (
            f"{row.home_display_name} vs {row.away_display_name} · "
            f"{pd.Timestamp(row['kickoff_utc']).tz_convert(ZoneInfo(display_timezone)).strftime('%b %d, %H:%M %Z')}"
        ),
        axis=1,
    )
    label_to_id = dict(zip(labels, ids))
    choice = st.selectbox("Fixture", labels.tolist(), index=default_index)
    fixture_id = label_to_id[choice]
    st.query_params["fixture"] = fixture_id
    return fixture_id


def load_score_matrix(context, fixture_id: str) -> np.ndarray:
    descriptor = context.repository.descriptors["score_matrices"]
    suffix = context.repository.path("score_matrices").suffix.lower()
    if suffix == ".npz":
        arrays = context.repository.arrays("score_matrices")
        for key in (fixture_id, fixture_id.replace(":", "__")):
            if key in arrays:
                return np.asarray(arrays[key], dtype=float)
        if "fixture_ids" in arrays and "matrices" in arrays:
            ids = arrays["fixture_ids"].astype(str).tolist()
            return np.asarray(arrays["matrices"][ids.index(fixture_id)], dtype=float)
        raise KeyError(f"No score matrix for {fixture_id}")
    long = context.repository.frame("score_matrices")
    selected = long.loc[long.fixture_id.astype(str) == fixture_id]
    if selected.empty:
        raise KeyError(f"No score matrix for {fixture_id}")
    maximum_home = int(selected.home_goals.max())
    maximum_away = int(selected.away_goals.max())
    matrix = np.zeros((maximum_home + 1, maximum_away + 1), dtype=float)
    matrix[
        selected.home_goals.astype(int).to_numpy(),
        selected.away_goals.astype(int).to_numpy(),
    ] = selected.probability.to_numpy(dtype=float)
    return matrix


def _match_header(fixture: pd.Series, forecast: pd.Series, display_timezone: str) -> None:
    kickoff = pd.Timestamp(fixture.kickoff_utc).tz_convert(ZoneInfo(display_timezone))
    st.title(f"{fixture.home_display_name} vs {fixture.away_display_name}")
    venue = fixture.get("venue_name", fixture.get("venue_id", "Venue pending"))
    st.caption(
        f"{kickoff.strftime('%A, %B %d · %H:%M %Z')} · {venue} · "
        f"issued {pd.Timestamp(forecast.get('issued_at')).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    home_id = quote(str(fixture.home_team_id), safe="")
    away_id = quote(str(fixture.away_team_id), safe="")
    st.markdown(
        f"[Open {html.escape(str(fixture.home_display_name))} team page](/teams?team={home_id})"
        f" · [Open {html.escape(str(fixture.away_display_name))} team page](/teams?team={away_id})"
    )


def _share_card(fixture: pd.Series, forecast: pd.Series, markets: dict) -> str:
    title = html.escape(f"{fixture.home_display_name} vs {fixture.away_display_name}")
    model = html.escape(str(forecast.get("model_id", "unknown")))
    issued = html.escape(str(forecast.get("issued_at", "unknown")))
    stability = float(forecast.get("leader_stability", 0.0))
    interval = (
        f"Home 80% {float(forecast.get('p_home_lower80', forecast.p_home)):.0%}–"
        f"{float(forecast.get('p_home_upper80', forecast.p_home)):.0%}"
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:16px system-ui;max-width:720px;margin:40px auto;padding:24px;border:1px solid #d0d5dd;border-radius:16px}}.p{{display:flex;gap:28px;font-size:24px}}small{{color:#475467}}</style>
</head><body><main><h1>{title}</h1><div class="p"><b>Home {float(forecast.p_home):.0%}</b><b>Draw {float(forecast.p_draw):.0%}</b><b>Away {float(forecast.p_away):.0%}</b></div>
<p>Most likely score: {markets['most_likely_score']} · Over 2.5: {float(markets['p_over_2_5']):.0%} · BTTS: {float(markets['p_btts_yes']):.0%}</p>
<p>{interval} · {'Stable' if stability >= 0.75 else 'Fragile'} leader ({stability:.0%} bootstrap agreement).</p>
<small>Model {model} · issued {issued}. Probabilistic forecast for responsible, informational use.</small></main></body></html>"""


def _scenario_lab(forecast: pd.Series, context) -> None:
    default_controls = (
        ScenarioControl("lineup_strength_delta", "Lineup strength", -1.0, 1.0, 0.1),
        ScenarioControl("rest_days_delta", "Extra rest days", -3.0, 7.0, 1.0),
        ScenarioControl("home_advantage_delta", "Home advantage", -1.0, 1.0, 0.1),
        ScenarioControl("weather_intensity", "Weather intensity", 0.0, 1.0, 0.1),
        ScenarioControl("tactical_pace_delta", "Tactical pace", -1.0, 1.0, 0.1),
    )
    adapter = context.scenario_adapter
    controls = tuple(adapter.controls) if adapter is not None else default_controls
    with st.expander("What-if scenario lab"):
        if not context.has_capability("scenario_inference"):
            st.info(
                "Scenario inference is not available for this consumer. Controls are "
                "disabled so the app never substitutes a heuristic for the deployed model."
            )
            columns = st.columns(2)
            for index, control in enumerate(controls):
                columns[index % 2].slider(
                    control.label,
                    control.minimum,
                    control.maximum,
                    0.0,
                    control.step,
                    disabled=True,
                    key=f"scenario_disabled_{control.feature}",
                )
            return
        if adapter is None:
            st.warning(
                "The manifest advertises scenario inference, but this application "
                "has no deployed ScenarioInferenceAdapter. Controls remain hidden."
            )
            return
        st.caption(
            "Controls are evaluated by the deployed model pipeline; results are "
            "sensitivities, not newly observed forecasts."
        )
        changes = {}
        columns = st.columns(2)
        for index, control in enumerate(controls):
            base_value = float(adapter.base_features.get(control.feature, 0.0))
            changes[control.feature] = columns[index % 2].slider(
                control.label,
                control.minimum,
                control.maximum,
                base_value,
                control.step,
                key=f"scenario_{control.feature}",
            )
        base = np.asarray(adapter.cached_probability, dtype=float)
        scenario_probability = adapter.predict(changes)
        display = pd.DataFrame({
            "Outcome": ["Home", "Draw", "Away"],
            "Base": base,
            "Scenario": scenario_probability,
            "Delta": scenario_probability - base,
        })
        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config={
                "Base": st.column_config.NumberColumn(format="percent"),
                "Scenario": st.column_config.NumberColumn(format="percent"),
                "Delta": st.column_config.NumberColumn(format="%+.1%"),
            },
        )


def render(context) -> None:
    fixtures = context.repository.frame("fixtures")
    forecasts = context.repository.frame("forecasts")
    scheduled = fixtures.loc[fixtures.status.astype(str).str.lower() == "scheduled"].copy()
    if scheduled.empty:
        st.info("No scheduled fixtures are available.")
        return
    fixture_id = selected_fixture(scheduled, context.display_timezone)
    fixture = fixtures.assign(_id=fixtures.fixture_id.astype(str)).set_index("_id").loc[fixture_id]
    forecast = forecasts.assign(_id=forecasts.fixture_id.astype(str)).set_index("_id").loc[fixture_id]
    _match_header(fixture, forecast, context.display_timezone)
    render_probability_header(forecast)
    matrix = load_score_matrix(context, fixture_id)
    markets = markets_from_score_matrix(matrix)
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Scoreline probabilities")
        render_score_matrix(
            matrix, str(fixture.home_display_name), str(fixture.away_display_name)
        )
    with right:
        st.subheader("Goal outlook")
        expected_home = sum(index * matrix[index, :].sum() for index in range(matrix.shape[0]))
        expected_away = sum(index * matrix[:, index].sum() for index in range(matrix.shape[1]))
        a, b, c = st.columns(3)
        a.metric("Expected goals", f"{expected_home:.2f}–{expected_away:.2f}")
        b.metric("BTTS", f"{float(markets['p_btts_yes']):.0%}")
        c.metric("Mode", str(markets["most_likely_score"]))
        ladder = goal_market_frame(markets)
        st.dataframe(
            ladder,
            hide_index=True,
            width="stretch",
            column_config={
                "Over": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent"),
                "Under": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent"),
            },
        )
    if context.repository.available("forecast_explanations"):
        drivers = context.repository.frame("forecast_explanations")
        drivers = drivers.loc[drivers.fixture_id.astype(str) == fixture_id]
        st.subheader("Why this forecast")
        explanation_models = sorted(set(
            drivers.get("model_id", pd.Series(dtype=str)).dropna().astype(str)
        ))
        explanation_versions = sorted(set(
            drivers.get("model_version", pd.Series(dtype=str)).dropna().astype(str)
        ))
        st.caption(
            "Explanation provenance: model "
            + (", ".join(explanation_models) or str(forecast.get("model_id", "unknown")))
            + (f" · producer {', '.join(explanation_versions)}" if explanation_versions else "")
        )
        for bullet in evidence_bullets(
            drivers, str(forecast.get("uncertainty_caveat", "Probability intervals reflect model and sample uncertainty."))
        ):
            st.markdown(f"- {bullet}")
        outcome = st.segmented_control(
            "Driver outcome", ["home", "draw", "away"], default="home"
        )
        render_drivers(drivers, outcome)
    if context.repository.available("forecast_ledger"):
        revisions = context.repository.frame("forecast_ledger")
        revisions = revisions.loc[revisions.fixture_id.astype(str) == fixture_id]
        if not revisions.empty:
            st.subheader("Forecast revision timeline")
            st.dataframe(revisions.sort_values("issued_at"), hide_index=True, width="stretch")
            labels = set(revisions.revision_label.astype(str)) if "revision_label" in revisions else set()
            required = {"initial", "24_hour", "lineup", "closing"}
            if required.issubset(labels):
                st.caption("Initial, 24-hour, lineup, and closing snapshots are all available.")
            else:
                st.warning("Missing forecast stages: " + ", ".join(sorted(required - labels)))
    context_rows = []
    for artifact in ("recovery_load", "manager_effects", "squad_availability", "referee_matchups"):
        if context.repository.available(artifact):
            frame = context.repository.frame(artifact)
            if "fixture_id" in frame:
                selected = frame.loc[frame.fixture_id.astype(str) == fixture_id].copy()
                if not selected.empty:
                    selected.insert(0, "context_type", artifact)
                    context_rows.append(selected)
    if context_rows:
        with st.expander("Observed pre-match context timeline"):
            st.dataframe(pd.concat(context_rows, ignore_index=True), hide_index=True, width="stretch")
    _scenario_lab(forecast, context)
    with st.expander("Context and provenance"):
        for name in ("weather", "squads", "managers", "referees", "travel"):
            if name in context.capabilities:
                render_capability({"name": name.title(), **context.capabilities[name]})
            else:
                st.caption(f"{name.title()}: provider unavailable; base forecast remains valid.")
    card = _share_card(fixture, forecast, markets)
    st.download_button(
        "Download forecast card",
        card,
        file_name=f"{fixture_id.replace(':', '-')}-forecast.html",
        mime="text/html",
    )
