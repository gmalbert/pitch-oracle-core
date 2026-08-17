from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import numpy as np
import pandas as pd
import pytest

from pitch_oracle_core.analytics.radars import build_fixture_radars
from pitch_oracle_core.analytics.comparison import head_to_head_context
from pitch_oracle_core.analytics.schedule import fixture_difficulty
from pitch_oracle_core.analytics.style_clusters import style_fingerprints
from pitch_oracle_core.competition.simulation import (
    SimulationFixture,
    simulate_season,
    validate_simulation_probabilities,
)
from pitch_oracle_core.competition.phases import (
    bracket_paths,
    generate_pool_fixtures,
    phase_at_issue,
    transition_phase,
)
from pitch_oracle_core.competition.stakes import conditioned_match_stakes
from pitch_oracle_core.competition.standings import calculate_table
from pitch_oracle_core.context.managers import attach_manager_at_kickoff
from pitch_oracle_core.context.referees import attach_referee_at_kickoff
from pitch_oracle_core.context.revisions import forecast_revision_deltas
from pitch_oracle_core.context.schedule import haversine_km
from pitch_oracle_core.context.schedule import recovery_features
from pitch_oracle_core.context.snapshots import Snapshot, latest_usable_snapshot
from pitch_oracle_core.context.squad import availability_at_kickoff
from pitch_oracle_core.data.validation import (
    require_quality_report_payload,
    validate_forecast_artifacts,
    validate_pre_match_features,
    validate_publication_bundle,
)
from pitch_oracle_core.data.providers import CapabilityStatus, load_optional_provider
from pitch_oracle_core.domain import (
    CompetitionEdition,
    CompetitionRules,
    PhaseRule,
    EntityResolver,
    ProbabilityGrid,
    Team,
    TeamAlias,
    feature_catalog_statuses,
    research_initiatives,
)
from pitch_oracle_core.domain.competitions import rules_from_league_config
from pitch_oracle_core.domain.catalog import validate_complete_catalog
from pitch_oracle_core.domain.catalog import manifest_feature_statuses
from pitch_oracle_core.evaluation.cohorts import assign_evaluation_cohorts
from pitch_oracle_core.evaluation.experiments import (
    ExperimentSpec,
    experiment_partition,
    write_experiment_log,
)
from pitch_oracle_core.evaluation.registry import evaluate_release_decision
from pitch_oracle_core.evaluation.rolling_origin import append_forecast_ledger
from pitch_oracle_core.domain.research import validate_research_registry
from pitch_oracle_core.evaluation.scores import (
    multiclass_brier,
    multiclass_log_loss,
    ranked_probability_score,
)
from pitch_oracle_core.events.value import action_features_at_kickoff
from pitch_oracle_core.features.ledger import add_prior_team_state, build_team_events
from pitch_oracle_core.markets.devig import DevigMethod, devig
from pitch_oracle_core.markets.settlement import (
    handicap_quarter_net_return,
    over_quarter_net_return,
    under_quarter_net_return,
)
from pitch_oracle_core.markets.portfolio import (
    StakePolicy,
    assess_market,
    allocate_portfolio,
    backtest_summary,
    bankroll_backtest,
    recommended_stake_fraction,
)
from pitch_oracle_core.markets.benchmark import (
    OddsSnapshot,
    closing_line_audit,
    infer_market_implied_goals,
)
from pitch_oracle_core.markets.quotes import (
    consensus_market,
    latest_quotes_before,
    market_movement,
    quote_status,
)
from pitch_oracle_core.models.distribution_registry import (
    bivariate_poisson_grid,
    diagonal_inflated_grid,
    negative_binomial_grid,
)
from pitch_oracle_core.models.dixon_coles import DixonColesForecaster, dc_tau
from pitch_oracle_core.models.hierarchical_prior import (
    cold_start_badge,
    transfer_strength_prior,
)
from pitch_oracle_core.models.independent_poisson import independent_poisson_grid
from pitch_oracle_core.models.elo import EloModel
from pitch_oracle_core.models.protocol import FixtureFeatures
from pitch_oracle_core.models.protocol import ForecastTrack, ModelSpec, validate_fixture_track
from pitch_oracle_core.models.rank_covariate import RankCovariateGoalsModel
from pitch_oracle_core.models.stacking import CalibratedEnsemble
from pitch_oracle_core.ui.navigation import PageSpec, enabled
from pitch_oracle_core.ui.components.drivers import evidence_bullets
from pitch_oracle_core.ui.pages.match_center import _share_card
from prepare_model_data import prepare_historical_features
from pitch_oracle_core.ui.scenarios import (
    ScenarioControl,
    ScenarioInferenceAdapter,
    apply_scenario,
)


def test_all_feature_and_research_ids_are_intentionally_implemented():
    feature_status = feature_catalog_statuses()
    research_status = research_initiatives()
    validate_complete_catalog(feature_status)
    validate_research_registry(research_status)
    assert {item.status for item in feature_status} <= {"implemented", "capability_gated"}
    for item in feature_status:
        for module in item.implementation_modules:
            importlib.import_module(module)
    assert all(item.implementation for item in research_status)


def test_consumer_feature_statuses_are_explicit_and_honest():
    rows = manifest_feature_statuses(
        available_artifacts={"fixtures", "forecasts", "score_matrices"},
        capabilities={"odds": "unavailable", "scenario_inference": "unavailable"},
    )
    assert len(rows) == 50
    assert {row["status"] for row in rows} <= {
        "shipped", "intentionally_deferred", "capability_unavailable"
    }
    assert all(row["reason"] for row in rows)
    assert next(row for row in rows if row["feature_id"] == "F49")["status"] == "capability_unavailable"


def test_rank_covariate_challenger_is_point_in_time_and_emits_a_grid():
    kickoff = pd.date_range("2026-01-01", periods=10, freq="7D", tz="UTC")
    matches = pd.DataFrame({
        "kickoff_utc": kickoff,
        "home_goals": [2, 1, 3, 0, 2, 1, 4, 1, 2, 0],
        "away_goals": [0, 1, 1, 2, 1, 0, 1, 2, 0, 1],
        "home_rank": [1, 3, 2, 6, 4, 5, 1, 6, 2, 4],
        "away_rank": [6, 4, 5, 1, 3, 2, 5, 2, 6, 1],
        "home_elo": [1600, 1510, 1570, 1450, 1530, 1500, 1620, 1460, 1580, 1490],
        "away_elo": [1450, 1490, 1470, 1600, 1510, 1550, 1480, 1580, 1440, 1610],
        "home_rest_days": [7, 6, 8, 5, 7, 7, 9, 5, 8, 6],
        "away_rest_days": [5, 7, 6, 8, 7, 5, 6, 8, 5, 9],
        "home_form_points": [12, 8, 11, 4, 9, 7, 13, 5, 12, 6],
        "away_form_points": [4, 7, 6, 12, 8, 10, 5, 11, 3, 13],
    })
    model = RankCovariateGoalsModel(alpha=0.5).fit(
        matches, cutoff_utc=datetime(2026, 3, 1, tzinfo=timezone.utc)
    )
    fixture = FixtureFeatures(
        "fx", datetime(2026, 3, 8, tzinfo=timezone.utc), "home", "away",
        {
            "rank_difference": -3.0,
            "elo_difference": 120.0,
            "rest_difference": 2.0,
            "form_difference": 5.0,
        },
    )
    grid = model.predict_grid(fixture)
    assert grid.represented_mass + grid.tail_mass == pytest.approx(1.0)
    assert np.isfinite(grid.mass).all()


def test_entity_resolution_covers_reviewed_belgium_aliases():
    names = {
        "Anderlecht": "anderlecht",
        "Antwerp": "antwerp",
        "Cercle Brugge KSV": "cercle-brugge",
        "Club Brugge": "club-brugge",
        "KAA Gent": "gent",
        "KV Kortrijk": "kortrijk",
        "KV Mechelen": "mechelen",
        "KVC Westerlo": "westerlo",
        "Lommel SK": "lommel",
        "OH Leuven": "oh-leuven",
        "RAAL La Louvière": "raal-la-louviere",
        "Racing Genk": "genk",
        "Royal Charleroi SC": "charleroi",
        "Sint-Truidense": "st-truiden",
        "Standard Liege": "standard-liege",
        "Union St.-Gilloise": "union-sg",
        "Waasland-Beveren": "beveren",
        "Zulte-Waregem": "waregem",
    }
    teams = [Team(f"bel:{slug}", display, "BEL") for display, slug in names.items()]
    aliases = [
        TeamAlias("espn", display, f"bel:{slug}")
        for display, slug in names.items()
        if display not in {team.canonical_name for team in teams[:2]}
    ]
    resolver = EntityResolver(teams, aliases)
    resolved = [resolver.resolve("espn", name, date(2026, 8, 10)) for name in names]
    assert all(item.team_id for item in resolved)
    assert len({item.team_id for item in resolved}) == 18


def test_winter_match_remains_in_august_start_edition():
    edition = CompetitionEdition(
        "bel.1:2026-27", "bel.1", "Belgian Pro League", "Europe/Brussels",
        8, (), "bel.1-2026-27-v1",
    )
    assert edition.season_id(datetime(2027, 2, 1, tzinfo=timezone.utc)) == "2026-27"


def test_away_match_updates_next_home_rest_and_form():
    matches = pd.DataFrame([
        {"fixture_id": "f1", "edition_id": "x:2026-27", "kickoff_utc": "2026-08-01T14:00Z",
         "home_team_id": "a", "away_team_id": "b", "home_goals": 0, "away_goals": 2},
        {"fixture_id": "f2", "edition_id": "x:2026-27", "kickoff_utc": "2026-08-08T14:00Z",
         "home_team_id": "b", "away_team_id": "c", "home_goals": 1, "away_goals": 1},
    ])
    events = add_prior_team_state(build_team_events(matches))
    b_next = events.loc[(events.fixture_id == "f2") & (events.team_id == "b")].iloc[0]
    assert b_next.rest_days == 7
    assert b_next.points_l5 == 3


def test_future_scheduled_fixture_does_not_count_as_form():
    matches = pd.DataFrame([
        {"fixture_id": "f1", "edition_id": "x", "kickoff_utc": "2026-08-01T14:00Z",
         "home_team_id": "a", "away_team_id": "b", "home_goals": 1, "away_goals": 0},
        {"fixture_id": "f2", "edition_id": "x", "kickoff_utc": "2026-08-08T14:00Z",
         "home_team_id": "a", "away_team_id": "c", "home_goals": np.nan, "away_goals": np.nan},
        {"fixture_id": "f3", "edition_id": "x", "kickoff_utc": "2026-08-15T14:00Z",
         "home_team_id": "a", "away_team_id": "d", "home_goals": np.nan, "away_goals": np.nan},
    ])
    events = add_prior_team_state(build_team_events(matches))
    third = events.loc[(events.fixture_id == "f3") & (events.team_id == "a")].iloc[0]
    assert third.points_l5 == 3
    assert third.history_n == 1


def test_shuffling_future_results_cannot_change_pre_kickoff_state():
    matches = pd.DataFrame([
        {"fixture_id": "past", "edition_id": "x", "kickoff_utc": "2026-01-01T12:00Z", "home_team_id": "a", "away_team_id": "b", "home_goals": 1, "away_goals": 0},
        {"fixture_id": "target", "edition_id": "x", "kickoff_utc": "2026-01-08T12:00Z", "home_team_id": "a", "away_team_id": "c", "home_goals": 0, "away_goals": 0},
        {"fixture_id": "future1", "edition_id": "x", "kickoff_utc": "2026-01-15T12:00Z", "home_team_id": "a", "away_team_id": "d", "home_goals": 5, "away_goals": 0},
        {"fixture_id": "future2", "edition_id": "x", "kickoff_utc": "2026-01-22T12:00Z", "home_team_id": "a", "away_team_id": "e", "home_goals": 0, "away_goals": 4},
    ])
    first = add_prior_team_state(build_team_events(matches))
    shuffled = matches.copy()
    shuffled.loc[shuffled.fixture_id == "future1", ["home_goals", "away_goals"]] = [0, 9]
    shuffled.loc[shuffled.fixture_id == "future2", ["home_goals", "away_goals"]] = [8, 0]
    second = add_prior_team_state(build_team_events(shuffled))
    columns = ["history_n", "points_l5", "rest_days"]
    first_row = first.loc[(first.fixture_id == "target") & (first.team_id == "a"), columns]
    second_row = second.loc[(second.fixture_id == "target") & (second.team_id == "a"), columns]
    pd.testing.assert_frame_equal(first_row.reset_index(drop=True), second_row.reset_index(drop=True))


def test_publication_gate_requires_point_in_time_metadata_and_coherence():
    fixtures = pd.DataFrame([{
        "fixture_id": "f", "edition_id": "x:2026-27", "rules_version": "x-v1",
        "kickoff_utc": "2026-08-10T18:00:00Z", "home_team_id": "a",
        "away_team_id": "b", "provider_event_id": "p1", "source": "test",
        "observed_at": "2026-08-09T12:00:00Z", "status": "scheduled",
    }])
    forecasts = pd.DataFrame([{
        "fixture_id": "f", "issued_at": "2026-08-10T12:00:00Z",
        "effective_sample_size": 42.0, "cold_start_status": "full_history",
        "p_home": 0.5, "p_draw": 0.2, "p_away": 0.3,
    }])
    scorelines = pd.DataFrame([
        {"fixture_id": "f", "home_goals": 1, "away_goals": 0, "probability": 0.5},
        {"fixture_id": "f", "home_goals": 0, "away_goals": 0, "probability": 0.2},
        {"fixture_id": "f", "home_goals": 0, "away_goals": 1, "probability": 0.3},
    ])
    features = pd.DataFrame([{
        "fixture_id": "f", "feature_timestamp": "2026-08-10T11:59:00Z",
        "rating_difference": 25.0,
    }])
    report = validate_publication_bundle(
        fixtures=fixtures, forecasts=forecasts, scorelines=scorelines,
        pre_match_features=features,
    )
    assert report.publishable
    late = features.assign(feature_timestamp="2026-08-10T18:00:00Z")
    assert not validate_pre_match_features(late, fixtures).publishable


def test_standings_apply_rules_and_adjustments():
    matches = pd.DataFrame([
        {"fixture_id": "f1", "edition_id": "x", "kickoff_utc": "2026-08-01T14:00Z",
         "home_team_id": "a", "away_team_id": "b", "home_goals": 2, "away_goals": 0},
        {"fixture_id": "f2", "edition_id": "x", "kickoff_utc": "2026-08-02T14:00Z",
         "home_team_id": "c", "away_team_id": "a", "home_goals": 0, "away_goals": 0},
    ])
    table = calculate_table(matches, CompetitionRules("v1", points_adjustments={"a": -1}))
    assert table.iloc[0].team_id == "a"
    assert table.iloc[0].points == 3


def test_golden_table_and_data_driven_belgium_scotland_phase_rules():
    from pitch_oracle_core import get_league_config

    matches = pd.DataFrame([
        {"fixture_id": "1", "edition_id": "x", "kickoff_utc": "2026-01-01T12:00Z", "home_team_id": "a", "away_team_id": "b", "home_goals": 2, "away_goals": 0},
        {"fixture_id": "2", "edition_id": "x", "kickoff_utc": "2026-01-02T12:00Z", "home_team_id": "c", "away_team_id": "d", "home_goals": 1, "away_goals": 1},
        {"fixture_id": "3", "edition_id": "x", "kickoff_utc": "2026-01-03T12:00Z", "home_team_id": "a", "away_team_id": "c", "home_goals": 0, "away_goals": 1},
        {"fixture_id": "4", "edition_id": "x", "kickoff_utc": "2026-01-04T12:00Z", "home_team_id": "b", "away_team_id": "d", "home_goals": 3, "away_goals": 0},
        {"fixture_id": "5", "edition_id": "x", "kickoff_utc": "2026-01-05T12:00Z", "home_team_id": "a", "away_team_id": "d", "home_goals": 1, "away_goals": 0},
        {"fixture_id": "6", "edition_id": "x", "kickoff_utc": "2026-01-06T12:00Z", "home_team_id": "b", "away_team_id": "c", "home_goals": 0, "away_goals": 0},
    ])
    golden = calculate_table(matches, CompetitionRules("golden-v1"))
    assert golden.team_id.tolist() == ["a", "c", "b", "d"]
    assert golden.points.tolist() == [6, 5, 4, 1]

    belgium = rules_from_league_config(
        get_league_config("belgium"), version="bel.1:2026-27:v1"
    )
    ranking = [f"t{i:02d}" for i in range(1, 19)]
    points = {team: 63 - index for index, team in enumerate(ranking)}
    belgium_transition = transition_phase(ranking, points, belgium.phases[1])
    assert list(belgium_transition.pools) == ["champions", "europe", "relegation"]
    assert belgium_transition.starting_points["t01"] == 32
    assert len(generate_pool_fixtures(belgium_transition)) == 45
    assert belgium.outcome_labels["champions_playoff"] == (1, 2, 3, 4, 5, 6)

    scotland = rules_from_league_config(
        get_league_config("scotland"), version="sco.1:2026-27:v1"
    )
    scottish_ranking = [f"s{i:02d}" for i in range(1, 13)]
    scottish_points = {team: 50 - index for index, team in enumerate(scottish_ranking)}
    split = transition_phase(scottish_ranking, scottish_points, scotland.phases[1])
    assert list(split.pools) == ["top_6", "bottom_6"]
    assert split.starting_points == {team: float(value) for team, value in scottish_points.items()}
    playoff = transition_phase(scottish_ranking, scottish_points, scotland.phases[2])
    assert bracket_paths(playoff)[0]["legs"] == 2


def test_probability_grid_and_all_primary_scores_are_coherent():
    grid = independent_poisson_grid(1.5, 1.0)
    assert grid.represented_mass + grid.tail_mass == pytest.approx(1.0)
    assert grid.normalized_one_x_two().sum() == pytest.approx(1.0)
    y = np.array([0, 1, 2])
    perfect = np.eye(3) * 0.999998 + (1 - np.eye(3)) * 0.000001
    uniform = np.full((3, 3), 1 / 3)
    assert multiclass_log_loss(y, perfect) < multiclass_log_loss(y, uniform)
    assert multiclass_brier(y, perfect) < multiclass_brier(y, uniform)
    assert ranked_probability_score(y, perfect) < ranked_probability_score(y, uniform)


def test_distribution_challengers_emit_finite_mass():
    grids = [
        bivariate_poisson_grid(1.6, 1.2, 0.15),
        negative_binomial_grid(1.6, 1.2, 3.0),
    ]
    grids.append(diagonal_inflated_grid(grids[0], 0.10))
    for grid in grids:
        assert np.isfinite(grid.mass).all()
        assert (grid.mass >= 0).all()
        assert grid.represented_mass + grid.tail_mass == pytest.approx(1.0)


def test_invalid_dependent_count_parameters_fail_and_dc_fallback_is_measured():
    with pytest.raises(ValueError, match="non-positive"):
        dc_tau(0, 0, 2.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="shared rate"):
        bivariate_poisson_grid(1.0, 1.0, 1.0)
    sparse = pd.DataFrame([{
        "kickoff_utc": "2026-01-01T12:00Z", "home_team_id": "a",
        "away_team_id": "b", "home_goals": 1, "away_goals": 0,
    }])
    forecaster = DixonColesForecaster().fit(
        sparse, cutoff_utc=datetime(2026, 2, 1, tzinfo=timezone.utc)
    )
    fixture = FixtureFeatures(
        "f", datetime(2026, 2, 2, tzinfo=timezone.utc), "a", "b", {}
    )
    grid = forecaster.predict_grid(fixture)
    assert grid.represented_mass + grid.tail_mass == pytest.approx(1.0)
    assert forecaster.fallback_rate == pytest.approx(1.0)


def test_modal_score_and_most_likely_outcome_are_independent_labels():
    from pitch_oracle_core.domain.forecasts import markets_from_score_matrix

    matrix = np.array([[0.25, 0.15], [0.20, 0.10], [0.20, 0.10]])
    markets = markets_from_score_matrix(matrix)
    assert markets["most_likely_score"] == "0-0"
    assert markets["p_home"] == pytest.approx(0.50)
    assert markets["p_home"] > markets["p_draw"]


@pytest.mark.parametrize("method", list(DevigMethod))
def test_all_devig_methods_produce_a_fair_market(method):
    fair = devig(np.array([2.10, 3.40, 3.60]), method)
    assert fair.probabilities.sum() == pytest.approx(1.0)
    assert (fair.probabilities > 0).all()


def test_multiplicative_devig_matches_a_golden_market():
    fair = devig(np.array([2.0, 3.0, 4.0]), DevigMethod.MULTIPLICATIVE)
    assert fair.probabilities == pytest.approx(
        np.array([6 / 13, 4 / 13, 3 / 13]), abs=1e-12
    )


@pytest.mark.parametrize(
    ("total", "line", "expected"),
    [(3, "2.25", "1.00"), (2, "2.25", "-0.50"), (2, "1.75", "0.50"), (1, "1.75", "-1.00")],
)
def test_quarter_total_settlement(total, line, expected):
    assert over_quarter_net_return(total, Decimal(line), Decimal("2.00")) == Decimal(expected)


def test_quarter_handicap_settlement():
    assert handicap_quarter_net_return(1, 1, Decimal("-0.25"), Decimal("2.00")) == Decimal("-0.50")
    assert handicap_quarter_net_return(2, 1, Decimal("-0.75"), Decimal("2.00")) == Decimal("0.50")


def test_snapshot_at_or_after_kickoff_is_never_selected():
    kickoff = datetime(2026, 8, 10, 15, tzinfo=timezone.utc)
    snapshots = [
        Snapshot("a", kickoff.replace(hour=12), kickoff.replace(hour=0), None, "p", "1", 1),
        Snapshot("a", kickoff, kickoff.replace(hour=0), None, "p", "2", 2),
    ]
    assert latest_usable_snapshot(snapshots, kickoff).payload == 1


def test_manager_referee_and_squad_rows_are_valid_at_kickoff():
    fixtures = pd.DataFrame([{
        "fixture_id": "f", "kickoff_utc": "2026-08-10T18:00:00Z",
        "home_team_id": "a", "away_team_id": "b",
    }])
    tenures = pd.DataFrame([
        {"team_id": "a", "manager_id": "old", "effective_from": "2026-01-01T00:00Z", "effective_to": None, "observed_at": "2026-08-01T00:00Z"},
        {"team_id": "a", "manager_id": "future-news", "effective_from": "2026-08-01T00:00Z", "effective_to": None, "observed_at": "2026-08-10T19:00Z"},
        {"team_id": "b", "manager_id": "away", "effective_from": "2026-01-01T00:00Z", "effective_to": None, "observed_at": "2026-08-01T00:00Z"},
    ])
    managers = attach_manager_at_kickoff(fixtures, tenures)
    assert managers.iloc[0].home_manager_id == "old"
    assignments = pd.DataFrame([
        {"fixture_id": "f", "referee_id": "known", "observed_at": "2026-08-10T12:00Z"},
        {"fixture_id": "f", "referee_id": "late", "observed_at": "2026-08-10T19:00Z"},
    ])
    assert attach_referee_at_kickoff(fixtures, assignments).iloc[0].referee_id == "known"
    availability = pd.DataFrame([
        {"fixture_id": "f", "player_id": "p", "team_id": "a", "availability_probability": 0.2, "observed_at": "2026-08-10T12:00Z"},
        {"fixture_id": "f", "player_id": "p", "team_id": "a", "availability_probability": 1.0, "observed_at": "2026-08-10T19:00Z"},
    ])
    selected = availability_at_kickoff(availability, fixtures)
    assert selected.iloc[0].availability_probability == pytest.approx(0.2)


def test_optional_provider_failure_preserves_the_base_payload():
    result = load_optional_provider(
        "squads", "test-provider",
        loader=lambda: (_ for _ in ()).throw(RuntimeError("offline")),
        fallback=lambda: {"base_forecast_valid": True},
    )
    assert result.payload == {"base_forecast_valid": True}
    assert result.capability.status == CapabilityStatus.FAILED
    assert "base forecast remains available" in result.capability.message


def test_event_value_features_are_lagged_before_target_kickoff():
    summaries = pd.DataFrame([
        {"fixture_id": "past", "team_id": "a", "action_value": 5.0, "actions": 10, "observed_at": "2026-01-01T14:00Z"},
        {"fixture_id": "target", "team_id": "a", "action_value": 999.0, "actions": 1, "observed_at": "2026-01-10T19:00Z"},
        {"fixture_id": "past-b", "team_id": "b", "action_value": 2.0, "actions": 5, "observed_at": "2026-01-02T14:00Z"},
    ])
    fixtures = pd.DataFrame([{
        "fixture_id": "target", "home_team_id": "a", "away_team_id": "b",
        "kickoff_utc": "2026-01-10T18:00Z",
    }])
    features = action_features_at_kickoff(summaries, fixtures)
    home = features.loc[features.side == "home"].iloc[0]
    assert home.history_actions == 10
    assert home.lagged_action_value_per_action == pytest.approx(0.5)
    assert home.feature_timestamp < pd.Timestamp("2026-01-10T18:00Z")


def test_haversine_is_symmetric_and_zero_at_same_point():
    assert haversine_km(50.0, 4.0, 50.0, 4.0) == pytest.approx(0.0)
    assert haversine_km(50.0, 4.0, 51.0, 5.0) == pytest.approx(
        haversine_km(51.0, 5.0, 50.0, 4.0)
    )


def test_forecast_artifact_validator_reconciles_matrix_and_1x2():
    matrix = independent_poisson_grid(1.2, 0.9).normalized_mass()
    scorelines = pd.DataFrame([
        {"fixture_id": "f", "home_goals": home, "away_goals": away, "probability": matrix[home, away]}
        for home in range(matrix.shape[0]) for away in range(matrix.shape[1])
    ])
    home = scorelines.loc[scorelines.home_goals > scorelines.away_goals, "probability"].sum()
    draw = scorelines.loc[scorelines.home_goals == scorelines.away_goals, "probability"].sum()
    away = 1 - home - draw
    validate_forecast_artifacts(
        pd.DataFrame([{"fixture_id": "f", "status": "scheduled"}]),
        pd.DataFrame([{"fixture_id": "f", "p_home": home, "p_draw": draw, "p_away": away}]),
        scorelines,
    )


def test_scenario_reset_is_exact_and_forbidden_mutation_fails():
    base = {"rest": 3.0, "immutable": 5.0}
    controls = [ScenarioControl("rest", "Rest", 0, 10, 1)]
    assert apply_scenario(base, {"rest": 3.0}, controls) == base
    with pytest.raises(ValueError, match="immutable"):
        apply_scenario(base, {"immutable": 0.0}, controls)


def test_scenario_adapter_uses_one_predictor_and_exact_cached_reset():
    base = {"rest": 3.0, "strength": 0.2}
    cached = np.array([0.45, 0.30, 0.25])

    def predict(features):
        home = 0.45 + 0.01 * (float(features["rest"]) - 3.0)
        return np.array([home, 0.30, 1.0 - home - 0.30])

    adapter = ScenarioInferenceAdapter(
        base, (ScenarioControl("rest", "Rest", 0, 10, 1),), predict, cached
    )
    assert np.array_equal(adapter.predict({"rest": 3.0}), cached)
    assert adapter.predict({"rest": 5.0}).sum() == pytest.approx(1.0)


def test_immutable_forecast_ledger_rejects_probability_rewrites():
    row = pd.DataFrame([{
        "fixture_id": "f", "issued_at": "2026-08-10T12:00Z",
        "kickoff_utc": "2026-08-10T18:00Z", "model_id": "m",
        "p_home": 0.5, "p_draw": 0.2, "p_away": 0.3,
    }])
    ledger = append_forecast_ledger(pd.DataFrame(), row)
    with pytest.raises(ValueError, match="immutable forecast field"):
        append_forecast_ledger(ledger, row.assign(p_home=0.4))
    finalized = append_forecast_ledger(ledger, row.assign(actual_outcome=0))
    with pytest.raises(ValueError, match="already finalized"):
        append_forecast_ledger(finalized, row.assign(actual_outcome=2))


def test_independent_track_rejects_market_derived_fixture_columns():
    fixture = FixtureFeatures(
        "f", datetime(2026, 8, 10, tzinfo=timezone.utc), "a", "b",
        {"elo_difference": 20.0, "closing_market_price": 2.2},
    )
    spec = ModelSpec("m", "test", ForecastTrack.INDEPENDENT, frozenset(), {})
    with pytest.raises(ValueError, match="market-derived"):
        validate_fixture_track(spec, fixture)


def test_required_evaluation_cohorts_and_release_gate_are_reproducible():
    rows = 40
    y = np.arange(rows) % 3
    weak = np.full((rows, 3), 1 / 3)
    strong = np.full((rows, 3), 0.05)
    strong[np.arange(rows), y] = 0.90
    base = pd.DataFrame({
        "fixture_id": [f"f{i}" for i in range(rows)],
        "kickoff_utc": pd.date_range("2026-01-01", periods=rows, freq="7D", tz="UTC"),
        "issued_at": pd.date_range("2025-12-31", periods=rows, freq="7D", tz="UTC"),
        "actual_outcome": y, "p_home": weak[:, 0], "p_draw": weak[:, 1],
        "p_away": weak[:, 2], "round_number": (np.arange(rows) % 10) + 1,
        "phase": "regular", "coverage_status": "full",
    })
    tagged = assign_evaluation_cohorts(base)
    assert {"season:first_five_matchdays", "favorite:balanced", "phase:regular"} <= set(tagged.cohort)
    challenger = base.assign(
        p_home=strong[:, 0], p_draw=strong[:, 1], p_away=strong[:, 2]
    )
    decision = evaluate_release_decision(
        base, challenger, champion_model_id="base", challenger_model_id="better",
        block_length=4, bootstrap_repetitions=200, calibration_ece_limit=0.2,
        coherence_passed=True, operations_passed=True,
    )
    assert decision.promote


def test_frozen_experiment_hashes_candidates_thresholds_and_hides_forward_rows(tmp_path):
    spec = ExperimentSpec(
        "exp-v1", "A challenger improves", datetime(2026, 1, 1, tzinfo=timezone.utc),
        ("bel.1",), ("independent",), ("base", "candidate"), "log_loss",
        ("brier",), ("draw",), datetime(2025, 6, 30, tzinfo=timezone.utc),
        datetime(2025, 7, 1, tzinfo=timezone.utc),
        datetime(2026, 6, 30, tzinfo=timezone.utc), "matchweek",
        "white_style_reality_check", {"relative_gain": 0.005},
    )
    rows = pd.DataFrame({
        "kickoff_utc": pd.to_datetime(["2025-06-01T12:00Z", "2025-08-01T12:00Z"]),
        "value": [1, 2],
    })
    assert experiment_partition(rows, spec, purpose="tuning").value.tolist() == [1]
    assert experiment_partition(rows, spec, purpose="forward_test").value.tolist() == [2]
    destination = write_experiment_log(
        tmp_path / "experiment.json", spec,
        {"base": {"passed": True}, "candidate": {"passed": False}},
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert len(payload["spec_hash"]) == 64
    assert [item["model_id"] for item in payload["candidates"]] == ["base", "candidate"]
    assert payload["thresholds"] == {"relative_gain": 0.005}
    with pytest.raises(ValueError, match="coverage differs"):
        write_experiment_log(tmp_path / "bad.json", spec, {"base": {}})


def test_ensemble_falls_back_when_oof_rows_are_insufficient():
    y = np.array([0, 1, 2, 0, 1, 2])
    predictions = {
        "good": np.eye(3)[y] * 0.8 + 0.2 / 3,
        "weak": np.full((len(y), 3), 1 / 3),
    }
    ensemble = CalibratedEnsemble(minimum_stacking_rows=100).fit(predictions, y)
    assert ensemble.mode == "best_component_fallback"
    assert ensemble.fallback_component == "good"
    assert ensemble.predict_proba(predictions) == pytest.approx(predictions["good"])


def test_transfer_prior_provenance_and_cold_start_badge():
    prior = transfer_strength_prior(
        league_attack_mean=1.2, league_defense_mean=1.2,
        lower_division_attack=1.5, lower_division_defense=1.0,
        lower_division_weight=0.5, country_coefficient_adjustment=0.05,
        squad_value_adjustment=0.10,
    )
    assert "translated_lower_division" in prior.provenance
    assert "country_coefficient" in prior.provenance
    assert "squad_value" in prior.provenance
    assert cold_start_badge(prior.effective_matches, full_history_threshold=10) == "promoted_or_partial_prior"


def test_portfolio_caps_and_backtest_report_cover_risk_and_calibration():
    opportunities = pd.DataFrame([
        {"fixture_id": "f", "team_id": "a", "league_id": "x", "model_probability": 0.60,
         "market_probability": 0.50, "decimal_price": 2.10, "quote_is_fresh": True,
         "uncertainty_passed": True, "calibration_passed": True, "executable": True},
        {"fixture_id": "f", "team_id": "a", "league_id": "x", "model_probability": 0.59,
         "market_probability": 0.49, "decimal_price": 2.10, "quote_is_fresh": True,
         "uncertainty_passed": True, "calibration_passed": True, "executable": True},
    ])
    policy = StakePolicy(max_bet_fraction=0.01, max_fixture_fraction=0.015)
    allocation = allocate_portfolio(opportunities, policy)
    assert allocation.stake_fraction.sum() <= policy.max_fixture_fraction + 1e-12
    backtest_input = allocation.assign(
        kickoff_utc=pd.to_datetime(["2026-01-01T12:00Z", "2026-01-08T12:00Z"]),
        won=[True, False], provider="book", season_id="2025-26",
        closing_line_value=[0.02, -0.01],
    )
    ledger = bankroll_backtest(backtest_input, policy)
    summary = backtest_summary(ledger, bootstrap_repetitions=100)
    assert {"turnover", "roi", "maximum_drawdown", "calibration_brier", "mean_closing_line_value"} <= set(summary)


def test_quote_freshness_consensus_outlier_and_issue_time_boundary():
    probabilities = pd.DataFrame([
        {"fixture_id": "f", "market": "1x2", "selection": "home", "bookmaker": "a", "fair_probability": 0.50, "decimal_price": 2.0, "observed_at": "2026-01-01T11:00Z"},
        {"fixture_id": "f", "market": "1x2", "selection": "home", "bookmaker": "b", "fair_probability": 0.51, "decimal_price": 1.98, "observed_at": "2026-01-01T11:01Z"},
        {"fixture_id": "f", "market": "1x2", "selection": "home", "bookmaker": "outlier", "fair_probability": 0.95, "decimal_price": 1.05, "observed_at": "2026-01-01T11:02Z"},
    ])
    assert consensus_market(probabilities).iloc[0].consensus_probability == pytest.approx(0.51)
    quotes = probabilities.assign(executable=True)
    before = latest_quotes_before(quotes, datetime(2026, 1, 1, 11, 2, tzinfo=timezone.utc))
    assert set(before.bookmaker) == {"a", "b"}
    status = quote_status(
        "2026-01-01T11:00:00Z", "2026-01-01T11:10:00Z",
        executable=True, maximum_age_seconds=900,
    )
    assert status.fresh and status.executable
    with pytest.raises(ValueError, match="after"):
        quote_status("2026-01-01T12:00Z", "2026-01-01T11:00Z", executable=True)


def test_capability_driven_page_enablement():
    class Repo:
        def available(self, name):
            return name == "fixtures"

    class Context:
        repository = Repo()
        def has_capability(self, name):
            return name == "odds"

    page = PageSpec("x", "x", "x", "x", lambda context: None, ("fixtures",), ("odds",))
    assert enabled(page, Context())


class _TinyRules:
    @dataclass
    class State:
        points: dict[str, int]
        remaining: int

    def initial_state(self, completed):
        return self.State({"a": 0, "b": 0}, 1)

    def apply_score(self, state, fixture, home_goals, away_goals):
        state.remaining -= 1
        if home_goals > away_goals:
            state.points["a"] += 3
        elif away_goals > home_goals:
            state.points["b"] += 3
        else:
            state.points["a"] += 1
            state.points["b"] += 1

    def next_fixtures(self, state):
        return []

    def is_complete(self, state):
        return state.remaining == 0

    def ranked_teams(self, state):
        return sorted(state.points, key=lambda team: (-state.points[team], team))

    def outcome_labels(self, state):
        return {"title": {self.ranked_teams(state)[0]}}


def test_simulation_is_seeded_and_position_probabilities_reconcile():
    matrix = np.array([[0.2, 0.1], [0.1, 0.6]])
    fixture = SimulationFixture("f", "a", "b", matrix)
    first = simulate_season(_TinyRules(), None, [fixture], simulations=200, seed=1)
    second = simulate_season(_TinyRules(), None, [fixture], simulations=200, seed=1)
    assert first == second
    validate_simulation_probabilities(first, 2)


def test_match_stakes_conditioning_uses_common_random_numbers():
    matrix = np.array([[0.2, 0.1], [0.1, 0.6]])
    fixture = SimulationFixture("f", "a", "b", matrix)
    report = conditioned_match_stakes(
        _TinyRules(), None, [fixture], focal_fixture_id="f",
        simulations=100, seed=12,
    )
    assert report["common_random_numbers"] is True
    assert set(report["conditional_outcomes"]) == {"home", "draw", "away"}


def test_team_event_optional_metrics_are_perspective_normalized_and_lagged():
    matches = pd.DataFrame({
        "fixture_id": ["f1", "f2"], "edition_id": ["e", "e"],
        "kickoff_utc": ["2026-01-01T12:00Z", "2026-01-08T12:00Z"],
        "home_team_id": ["a", "b"], "away_team_id": ["b", "a"],
        "home_goals": [2, 0], "away_goals": [1, 1],
        "home_xg": [1.5, 0.8], "away_xg": [0.7, 1.2],
        "home_shots": [12, 8], "away_shots": [7, 10],
    })
    events = add_prior_team_state(build_team_events(matches))
    team_a = events.loc[events.team_id == "a"].sort_values("kickoff_utc")
    assert pd.isna(team_a.iloc[0].xg_for_ewm10)
    assert team_a.iloc[1].xg_for_ewm10 == pytest.approx(1.5)
    assert team_a.iloc[1].shots_for_ewm10 == pytest.approx(12.0)
    assert team_a.iloc[1].finishing_vs_expectation_ewm10 == pytest.approx(0.5)


def test_radar_filters_uncertain_upsets_and_blends_draw_and_event_inputs():
    frame = pd.DataFrame({
        "fixture_id": ["stable", "fragile"],
        "p_home": [0.45, 0.45], "p_draw": [0.30, 0.30], "p_away": [0.25, 0.25],
        "expected_total_goals": [3.0, 2.0], "p_btts_yes": [0.7, 0.4],
        "home_strength_probability": [0.35, 0.35],
        "away_strength_probability": [0.65, 0.65],
        "leader_stability": [0.85, 0.55],
        "p_home_lower80": [0.40, 0.20], "p_home_upper80": [0.50, 0.70],
        "p_draw_lower80": [0.27, 0.15], "p_draw_upper80": [0.33, 0.55],
        "p_away_lower80": [0.22, 0.10], "p_away_upper80": [0.28, 0.50],
        "calibrated_p_draw": [0.32, 0.28], "score_entropy": [2.5, 1.6],
        "shots_tempo": [24, 15], "style_matchup": [0.8, 0.2],
    })
    result = build_fixture_radars(frame).set_index("fixture_id")
    assert bool(result.loc["stable", "uncertainty_passed"])
    assert pd.notna(result.loc["stable", "upset_index"])
    assert not bool(result.loc["fragile", "uncertainty_passed"])
    assert pd.isna(result.loc["fragile", "upset_index"])
    assert result.loc["stable", "draw_calibration_gap"] == pytest.approx(0.02)
    assert result.loc["stable", "goal_fest_percentile"] > result.loc["fragile", "goal_fest_percentile"]


def test_elo_time_travel_uses_strict_pre_cutoff_history():
    model = EloModel()
    first = datetime(2026, 1, 1, tzinfo=timezone.utc)
    second = datetime(2026, 1, 8, tzinfo=timezone.utc)
    rating_after_first, _ = model.update("a", "b", 2, 0, observed_at=first)
    rating_before_second = model.rating_at("a", second)
    model.update("a", "b", 0, 2, observed_at=second)
    assert rating_before_second == pytest.approx(rating_after_first)
    assert model.rating_at("a", second) == pytest.approx(rating_after_first)
    with pytest.raises(ValueError, match="increasing"):
        model.update("a", "b", 1, 1, observed_at=second)


def test_head_to_head_counts_role_reversed_meetings_from_team_perspective():
    matches = pd.DataFrame({
        "fixture_id": ["a-home", "b-home"], "edition_id": ["e", "e"],
        "kickoff_utc": ["2025-01-01T12:00Z", "2025-06-01T12:00Z"],
        "home_team_id": ["a", "b"], "away_team_id": ["b", "a"],
        "home_goals": [1, 0], "away_goals": [0, 2],
    })
    context = head_to_head_context(
        build_team_events(matches), "a", "b",
        as_of="2026-01-01T00:00Z",
    )
    assert context["matches"] == 2
    assert set(context["meetings"].venue_role) == {"home", "away"}
    assert context["weighted_points_per_match"] == pytest.approx(3.0)


def test_revision_stages_are_initial_24_hour_lineup_and_closing():
    kickoff = pd.Timestamp("2026-01-10T12:00Z")
    ledger = pd.DataFrame({
        "fixture_id": ["f"] * 4, "kickoff_utc": [kickoff] * 4,
        "issued_at": [
            kickoff - pd.Timedelta(hours=168), kickoff - pd.Timedelta(hours=24),
            kickoff - pd.Timedelta(hours=2), kickoff - pd.Timedelta(minutes=6),
        ],
        "model_id": ["m"] * 4, "p_home": [.4, .41, .42, .43],
        "p_draw": [.3, .3, .3, .3], "p_away": [.3, .29, .28, .27],
    })
    assert forecast_revision_deltas(ledger).revision_label.tolist() == [
        "initial", "24_hour", "lineup", "closing"
    ]


def test_serialized_quality_report_blocks_manifest_publication():
    with pytest.raises(RuntimeError, match="alias_gaps"):
        require_quality_report_payload({
            "publishable": True,
            "checks": [{
                "check": "alias_gaps", "status": "failed", "severity": "blocking"
            }],
        })
    require_quality_report_payload({
        "publishable": True,
        "checks": [{
            "check": "alias_gaps", "status": "passed", "severity": "blocking"
        }],
    })


def test_phase_at_issue_uses_only_completed_rounds_before_cutoff():
    rules = CompetitionRules(
        "bel-v1",
        phases=(
            PhaseRule("regular"),
            PhaseRule("split", starts_after_round=30, pool_sizes=(6, 6, 4)),
        ),
    )
    fixtures = pd.DataFrame({
        "kickoff_utc": ["2026-03-01T12:00Z", "2026-03-08T12:00Z"],
        "round": [29, 30], "status": ["completed", "completed"],
    })
    assert phase_at_issue(
        fixtures, rules, datetime(2026, 3, 8, 11, tzinfo=timezone.utc)
    ) == "regular"
    assert phase_at_issue(
        fixtures, rules, datetime(2026, 3, 8, 13, tzinfo=timezone.utc)
    ) == "split"


def test_market_implied_goals_publishes_residual_instead_of_exact_claim():
    result = infer_market_implied_goals(
        fixture_id="f", issued_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        one_x_two=np.array([0.52, 0.27, 0.21]), over_2_5=0.58,
        expected_goal_difference=0.55,
        source_markets=("1x2", "total_2_5", "handicap"),
        devig_method="power",
    )
    assert result.expected_home > result.expected_away > 0
    assert result.solver_error >= 0
    assert result.source_markets == ("1x2", "total_2_5", "handicap")


def test_closing_line_audit_retains_source_time_coverage_and_devig_method():
    accepted = OddsSnapshot(
        "f", "1x2", "book", datetime(2026, 1, 1, 10, tzinfo=timezone.utc),
        (2.1, 3.4, 3.6), "open", True,
    )
    closing = OddsSnapshot(
        "f", "1x2", "book", datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
        (2.0, 3.5, 3.8), "close", True,
    )
    audit = closing_line_audit(
        accepted, closing, selection_index=0,
        devig_method=DevigMethod.POWER, source_coverage=0.8,
    )
    assert audit.clv > 0
    assert audit.devig_method == "power"
    assert audit.source_coverage == pytest.approx(0.8)


def test_market_movement_retains_opener_current_dispersion_and_books():
    rows = pd.DataFrame([
        {"fixture_id": "f", "market": "1x2", "selection": "home", "bookmaker": "a", "observed_at": "2026-01-01T10:00Z", "fair_probability": .48, "decimal_price": 2.1},
        {"fixture_id": "f", "market": "1x2", "selection": "home", "bookmaker": "b", "observed_at": "2026-01-01T10:00Z", "fair_probability": .50, "decimal_price": 2.0},
        {"fixture_id": "f", "market": "1x2", "selection": "home", "bookmaker": "a", "observed_at": "2026-01-01T12:00Z", "fair_probability": .52, "decimal_price": 1.95},
        {"fixture_id": "f", "market": "1x2", "selection": "home", "bookmaker": "b", "observed_at": "2026-01-01T12:00Z", "fair_probability": .54, "decimal_price": 1.90},
    ])
    movement = market_movement(rows).iloc[0]
    assert movement.opening_probability == pytest.approx(.49)
    assert movement.current_probability == pytest.approx(.53)
    assert movement.probability_move == pytest.approx(.04)
    assert movement.current_books == 2


def test_safety_gates_default_to_zero_and_backtest_deducts_costs():
    assessment = assess_market(.6, .5, 2.1)
    policy = StakePolicy()
    for gates in (
        (False, True, True, True), (True, False, True, True),
        (True, True, False, True), (True, True, True, False),
    ):
        assert recommended_stake_fraction(
            assessment, policy, quote_is_fresh=gates[0],
            uncertainty_passed=gates[1], calibration_passed=gates[2],
            executable=gates[3],
        ) == 0
    opportunity = pd.DataFrame([{
        "fixture_id": "f", "kickoff_utc": "2026-01-02T12:00Z",
        "model_probability": .6, "market_probability": .5,
        "decimal_price": 2.1, "quote_is_fresh": True,
        "uncertainty_passed": True, "calibration_passed": True,
        "executable": True, "won": True,
    }])
    ledger = bankroll_backtest(
        opportunity, policy, transaction_cost_fraction=.02
    )
    assert ledger.iloc[0].transaction_cost > 0
    assert ledger.iloc[0].profit < ledger.iloc[0].gross_profit


def test_quarter_settlement_is_bounded_across_full_score_and_line_grid():
    for goal_difference in range(-10, 11):
        home, away = max(goal_difference, 0), max(-goal_difference, 0)
        for quarter in range(-20, 21):
            line = Decimal(quarter) / Decimal(4)
            returned = handicap_quarter_net_return(home, away, line, Decimal("2.00"))
            assert Decimal("-1") <= returned <= Decimal("1")
    for total in range(0, 11):
        for quarter in range(0, 21):
            line = Decimal(quarter) / Decimal(4)
            over = over_quarter_net_return(total, line, Decimal("2.00"))
            under = under_quarter_net_return(total, line, Decimal("2.00"))
            assert over == -under


def test_cached_runtime_import_excludes_training_and_provider_frameworks():
    code = (
        "import json,sys; import pitch_oracle_core; "
        "import pitch_oracle_core.models.independent_poisson; "
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name.split('.')[0] in {'torch','sklearn','scipy','xgboost','shap','openmeteo_requests'})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert json.loads(result.stdout) == []


def test_runtime_extra_is_cache_only_and_legacy_app_shell_is_removed():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = " ".join(project["project"]["optional-dependencies"]["runtime"]).lower()
    assert not any(name in runtime for name in ("torch", "sklearn", "scipy", "xgboost", "shap"))
    assert not (root / "app_shell.py").exists()


def test_evidence_text_and_share_card_are_deterministic_and_accessible():
    drivers = pd.DataFrame([
        {"display_name": "Base strength", "explanation": "Rating edge.", "contribution": .2},
        {"display_name": "Rest", "explanation": "Two extra days.", "contribution": .1},
    ])
    first = evidence_bullets(drivers, "Interval remains material.")
    assert first == evidence_bullets(drivers, "Interval remains material.")
    assert first[-1] == "**Uncertainty:** Interval remains material."
    fixture = pd.Series({"home_display_name": "Home", "away_display_name": "Away"})
    forecast = pd.Series({
        "model_id": "m:v1", "issued_at": "2026-01-01T10:00Z",
        "p_home": .5, "p_draw": .3, "p_away": .2,
        "p_home_lower80": .42, "p_home_upper80": .58,
        "leader_stability": .8,
    })
    card = _share_card(fixture, forecast, {
        "most_likely_score": "1–0", "p_over_2_5": .48, "p_btts_yes": .44,
    })
    assert '<html lang="en">' in card
    assert 'name="viewport"' in card
    assert "responsible, informational use" in card
    assert "Home 80% 42%–58%" in card


def test_fixture_difficulty_rejects_post_kickoff_ratings():
    fixtures = pd.DataFrame([{
        "fixture_id": "f", "team_id": "a", "opponent_id": "b",
        "venue_role": "home", "kickoff_utc": "2026-01-02T12:00Z",
    }])
    ratings = pd.DataFrame([{
        "fixture_id": "f", "team_id": "b", "pre_match_rating": 1550,
        "observed_at": "2026-01-02T12:01Z",
    }])
    with pytest.raises(ValueError, match="post-kickoff"):
        fixture_difficulty(fixtures, ratings)
    result = fixture_difficulty(
        fixtures, ratings.assign(observed_at="2026-01-02T11:59Z")
    )
    assert result.iloc[0].difficulty_rating == pytest.approx(1495)


def test_low_confidence_venue_omits_travel_and_style_definitions_are_published():
    schedules = pd.DataFrame([
        {"team_id": "a", "kickoff_utc": "2026-01-01T12:00Z", "previous_venue_lat": 50., "previous_venue_lon": 4., "venue_lat": 51., "venue_lon": 5., "previous_venue_confidence": "verified", "venue_confidence": "verified"},
        {"team_id": "a", "kickoff_utc": "2026-01-08T12:00Z", "previous_venue_lat": 50., "previous_venue_lon": 4., "venue_lat": 52., "venue_lon": 6., "previous_venue_confidence": "verified", "venue_confidence": "low"},
    ])
    recovery = recovery_features(schedules)
    assert pd.notna(recovery.iloc[0].travel_km)
    assert pd.isna(recovery.iloc[1].travel_km)
    styles = style_fingerprints(pd.DataFrame({
        "team_id": ["a", "b"], "goals_for": [2., 1.],
        "goals_against": [1., 2.], "shots": [14., 8.],
    }))
    assert styles.cluster_stability.eq(1.0).all()
    assert styles.definition.str.contains("league-percentile").all()


def test_stale_optional_provider_is_excluded_from_inference_payload():
    observed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = load_optional_provider(
        "weather", "test", lambda: {"fresh_feature": 1}, lambda: {"base": 1},
        observed_at=observed, maximum_age_seconds=3600,
        now=datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
    )
    assert result.capability.status == CapabilityStatus.STALE
    assert result.payload == {"base": 1}
    assert "base forecast" in result.capability.message


def test_explicit_preparation_pipeline_uses_full_team_sequence_and_edition(tmp_path):
    source = tmp_path / "combined_historical_data.csv"
    destination = tmp_path / "features.csv"
    pd.DataFrame([
        {"Date": "01/08/2026", "Time": "14:00", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 0, "FTAG": 2, "FTR": "A"},
        {"Date": "08/08/2026", "Time": "14:00", "HomeTeam": "B", "AwayTeam": "C", "FTHG": 1, "FTAG": 1, "FTR": "D"},
        {"Date": "01/02/2027", "Time": "14:00", "HomeTeam": "B", "AwayTeam": "A", "FTHG": 1, "FTAG": 0, "FTR": "H"},
    ]).to_csv(source, sep="\t", index=False)
    result = prepare_historical_features(
        league_key="belgium", source=source, destination=destination
    )
    second = result.loc[result.fixture_id.str.contains("20260808")].iloc[0]
    assert second.HomeTeamPointsLast5 == pytest.approx(3.0)
    assert second.HomeRestDays == pytest.approx(7.0)
    assert set(result.Season) == {"2026-27"}
    assert result.kickoff_utc.map(lambda value: pd.Timestamp(value).tzinfo is not None).all()
    assert destination.is_file()


def test_fixture_ingestion_preserves_utc_and_rejects_missing_provider_ids(
    tmp_path, monkeypatch
):
    import fetch_upcoming_fixtures as fixture_fetcher

    payload = {
        "events": [
            {
                "id": "espn-1",
                "date": "2026-08-20T18:00:00Z",
                "status": {"type": {"name": "STATUS_SCHEDULED"}},
                "competitions": [{
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Anderlecht"}},
                        {"homeAway": "away", "team": {"displayName": "Club Brugge"}},
                    ]
                }],
            },
            {
                "date": "2026-08-21T18:00:00Z",
                "competitions": [],
            },
        ]
    }

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setattr(fixture_fetcher.requests, "get", lambda *args, **kwargs: Response())
    result = fixture_fetcher.fetch_upcoming_fixtures(
        "belgium", output_dir=tmp_path
    )
    assert result.fixture_id.tolist() == ["belgium:espn:espn-1"]
    assert result.provider_event_id.tolist() == ["espn-1"]
    assert result.edition_id.tolist() == ["bel.1:2026-27"]
    assert pd.Timestamp(result.kickoff_utc.iloc[0]).tz_convert("UTC").hour == 18
    assert result.Time.tolist() == ["20:00"]
