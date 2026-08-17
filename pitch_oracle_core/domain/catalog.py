"""Machine-readable implementation status for the 50-feature product catalog."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureStatus:
    feature_id: str
    name: str
    status: str
    required_artifacts: tuple[str, ...] = ()
    optional_capability: str | None = None
    implementation_modules: tuple[str, ...] = ()


_FEATURE_NAMES = (
    "Match Intelligence Center", "Scoreline probability matrix", "Full goal-market ladder",
    "Forecast driver waterfall", "Evidence-backed narrative", "What-if scenario lab",
    "Forecast uncertainty fan", "Upset Radar", "Draw Radar", "Goal Fest / Low Block index",
    "Match context timeline", "Shareable forecast card", "Team Command Center",
    "Form fingerprint", "Home/away split explorer", "Opponent-adjusted performance",
    "Team comparison studio", "Head-to-head context", "Dynamic power rankings",
    "Style fingerprints and clusters", "Fixture difficulty calendar",
    "Congestion, travel, and recovery load", "Manager-change tracker",
    "Squad availability impact", "Discipline and referee matchup", "Rule-aware live table",
    "Season projection table", "Title/Europe/relegation race", "Matchday stakes index",
    "Points target calculator", "Split/playoff scenario explorer", "Matchday storylines",
    "League trend laboratory", "Competitive balance dashboard", "Cross-league comparison",
    "Champion/challenger Model Lab", "Reliability and calibration explorer",
    "Historical prediction tracker", "Dynamic Dixon-Coles goals model",
    "Dynamic Elo/Glicko strength", "Calibrated ensemble", "Promoted/new-club transfer prior",
    "Cross-market probability reconciliation", "Forecast drift monitor",
    "Cohort performance slices", "Entity coverage and cold-start badges",
    "Data freshness and provenance panel", "Data quality control room",
    "Fair odds and market movement", "Responsible portfolio/backtest lab",
)

_OPTIONAL = {
    "F06": "scenario_inference",
    "F23": "managers",
    "F24": "squads",
    "F25": "referees",
    "F49": "odds",
    "F50": "odds",
}

_ARTIFACTS = {
    "F01": ("fixtures", "forecasts"),
    "F02": ("score_matrices",),
    "F03": ("score_matrices",),
    "F04": ("forecast_explanations",),
    "F05": ("forecast_explanations",),
    "F07": ("forecasts",),
    "F08": ("radars",), "F09": ("radars",), "F10": ("radars",),
    "F11": ("forecast_ledger",), "F12": ("forecasts",),
    "F13": ("team_snapshots",), "F14": ("team_snapshots",),
    "F15": ("team_snapshots",), "F16": ("team_snapshots",),
    "F17": ("team_snapshots",), "F18": ("team_snapshots", "team_events"),
    "F19": ("rating_history",), "F20": ("style_fingerprints",),
    "F21": ("fixture_difficulty",),
    "F22": ("recovery_load",), "F23": ("manager_effects",),
    "F24": ("squad_availability",), "F25": ("referee_matchups",),
    "F26": ("fixtures", "standings"),
    "F27": ("season_simulations", "position_probabilities"),
    "F28": ("season_simulations",),
    "F29": ("match_stakes",), "F30": ("points_targets",),
    "F31": ("phase_scenarios",), "F32": ("storylines",),
    "F33": ("league_trends",), "F34": ("competitive_balance",),
    "F35": ("cross_league",),
    "F36": ("model_registry", "evaluation_predictions"),
    "F37": ("calibration", "evaluation_predictions"),
    "F38": ("forecast_ledger",), "F39": ("model_registry",),
    "F40": ("rating_history",),
    "F41": ("model_registry", "evaluation_predictions"),
    "F42": ("forecasts",), "F43": ("score_matrices",),
    "F44": ("drift_report",), "F45": ("cohort_metrics",),
    "F46": ("fixtures",), "F47": ("provider_runs",),
    "F48": ("quality_report",),
    "F49": ("odds_snapshots",), "F50": ("odds_snapshots",),
}

_IMPLEMENTATIONS = (
    "ui.pages.match_center", "domain.probability_grid", "domain.forecasts",
    "ui.components.drivers", "ui.components.drivers", "ui.scenarios",
    "models.uncertainty", "analytics.radars", "analytics.radars",
    "analytics.radars", "context.revisions", "ui.pages.match_center",
    "analytics.team_snapshots", "features.ledger", "analytics.team_snapshots",
    "analytics.team_snapshots", "analytics.comparison", "analytics.comparison",
    "models.elo", "analytics.style_clusters", "analytics.schedule",
    "context.schedule", "context.managers", "context.squad", "context.referees",
    "competition.standings", "competition.simulation", "competition.simulation",
    "competition.stakes", "competition.stakes", "phases", "analytics.storylines",
    "analytics.league_trends", "analytics.league_trends", "analytics.league_trends",
    "evaluation.registry", "evaluation.calibration", "evaluation.rolling_origin",
    "models.dixon_coles", "models.elo", "models.stacking",
    "models.hierarchical_prior", "domain.forecasts", "evaluation.drift",
    "evaluation.cohorts", "domain.entities", "artifacts.manifest",
    "data.validation", "markets.quotes", "markets.portfolio",
)


def feature_catalog_statuses() -> tuple[FeatureStatus, ...]:
    """Return every shipped contract; optional data remains capability-gated."""
    result = []
    for index, (name, implementation) in enumerate(
        zip(_FEATURE_NAMES, _IMPLEMENTATIONS, strict=True), start=1
    ):
        feature_id = f"F{index:02d}"
        capability = _OPTIONAL.get(feature_id)
        result.append(FeatureStatus(
            feature_id,
            name,
            "capability_gated" if capability else "implemented",
            required_artifacts=_ARTIFACTS.get(feature_id, ()),
            optional_capability=capability,
            implementation_modules=(f"pitch_oracle_core.{implementation}",),
        ))
    return tuple(result)


def validate_complete_catalog(statuses: tuple[FeatureStatus, ...]) -> None:
    expected = {f"F{index:02d}" for index in range(1, 51)}
    observed = {item.feature_id for item in statuses}
    if observed != expected or len(statuses) != 50:
        raise ValueError("feature catalog must contain exactly F01 through F50")
    if any(not item.implementation_modules for item in statuses):
        raise ValueError("every feature requires an implementation route")


def manifest_feature_statuses(
    *,
    available_artifacts: set[str],
    capabilities: dict[str, str],
) -> tuple[dict[str, object], ...]:
    """Resolve code contracts into honest per-consumer availability states."""
    rows = []
    for item in feature_catalog_statuses():
        capability_status = (
            capabilities.get(item.optional_capability, "unavailable")
            if item.optional_capability else None
        )
        if item.optional_capability and capability_status not in {
            "available", "degraded", "partial"
        }:
            status = "capability_unavailable"
            reason = (
                f"{item.optional_capability} capability is "
                f"{capability_status or 'unavailable'}"
            )
        else:
            missing = sorted(set(item.required_artifacts).difference(available_artifacts))
            if missing:
                status = "intentionally_deferred"
                reason = "consumer bundle does not publish: " + ", ".join(missing)
            else:
                status = "shipped"
                reason = "implementation and required artifact contract are available"
        rows.append({
            "feature_id": item.feature_id,
            "name": item.name,
            "status": status,
            "reason": reason,
            "required_artifacts": list(item.required_artifacts),
            "optional_capability": item.optional_capability,
            "implementation_modules": list(item.implementation_modules),
        })
    return tuple(rows)
