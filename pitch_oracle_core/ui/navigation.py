"""Capability and artifact-driven native Streamlit navigation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import streamlit as st


@dataclass(frozen=True)
class PageSpec:
    group: str
    title: str
    icon: str
    path: str
    render: Callable
    required_artifacts: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()


def enabled(page: PageSpec, context) -> bool:
    return (
        all(context.repository.available(name) for name in page.required_artifacts)
        and all(context.has_capability(name) for name in page.required_capabilities)
    )


def page_specs() -> tuple[PageSpec, ...]:
    from .pages import (
        comparison, data_control, league_lab, market_lab, match_center, model_lab,
        overview, prediction_history, projections, radars, research_lab, standings,
        team_center,
    )

    return (
        PageSpec("", "Overview", ":material/home:", "overview", overview.render),
        PageSpec("Match Center", "Fixture Explorer", ":material/sports_soccer:", "match-center", match_center.render, ("fixtures", "forecasts", "score_matrices")),
        PageSpec("Match Center", "Fixture Radars", ":material/radar:", "radars", radars.render, ("radars",)),
        PageSpec("Match Center", "Prediction History", ":material/history:", "prediction-history", prediction_history.render, ("forecast_ledger",)),
        PageSpec("Teams", "Team Command Center", ":material/shield:", "teams", team_center.render, ("team_snapshots",)),
        PageSpec("Teams", "Comparison Studio", ":material/compare_arrows:", "comparison", comparison.render, ("team_snapshots",)),
        PageSpec("League", "Live Table", ":material/leaderboard:", "standings", standings.render, ("fixtures",)),
        PageSpec("League", "Season Projections", ":material/finance:", "projections", projections.render, ("season_simulations",)),
        PageSpec("League", "League Laboratory", ":material/query_stats:", "league-lab", league_lab.render),
        PageSpec("Models & Data", "Model Lab", ":material/model_training:", "model-lab", model_lab.render, ("model_registry",)),
        PageSpec("Models & Data", "Research Lab", ":material/science:", "research-lab", research_lab.render, ("research_experiments", "research_metrics", "research_calibration")),
        PageSpec("Models & Data", "Data Control Room", ":material/fact_check:", "data-control", data_control.render),
        PageSpec("Models & Data", "Market Lab", ":material/candlestick_chart:", "market-lab", market_lab.render, ("odds_snapshots",), ("odds",)),
    )


def build_navigation(context, specs: tuple[PageSpec, ...] | None = None):
    groups: dict[str, list] = {}
    for spec in specs or page_specs():
        if not enabled(spec, context):
            continue

        def renderer(active=spec):
            active.render(context)

        groups.setdefault(spec.group, []).append(st.Page(
            renderer,
            title=spec.title,
            icon=spec.icon,
            url_path=spec.path,
            default=spec.path == "overview",
        ))
    return st.navigation(groups)
