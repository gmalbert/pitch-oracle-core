"""Build a deterministic manifest-v3 demo bundle for UI/browser smoke testing."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitch_oracle_core import __version__, get_league_config
from pitch_oracle_core.analytics.radars import build_fixture_radars
from pitch_oracle_core.artifacts.manifest import (
    ManifestV3,
    descriptor_for_file,
    publish_manifest,
)
from pitch_oracle_core.domain.catalog import manifest_feature_statuses
from pitch_oracle_core.domain.forecasts import markets_from_score_matrix
from pitch_oracle_core.domain.research import research_initiatives
from pitch_oracle_core.data.validation import require_quality_report_payload
from pitch_oracle_core.evaluation.scores import score_panel
from pitch_oracle_core.evaluation.experiments import ExperimentSpec, write_experiment_log
from pitch_oracle_core.models.independent_poisson import independent_poisson_grid


def _frame(path: Path, rows: list[dict]) -> int:
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)
    return len(frame)


def build(root: Path, league_key: str = "belgium") -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_dir = root / "precomputed"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc)
    release_name = generated.strftime("%Y%m%dT%H%M%S%fZ")
    release_relative = f"precomputed/releases/{release_name}"
    precomputed = root / release_relative
    precomputed.mkdir(parents=True, exist_ok=False)
    config = get_league_config(league_key)
    competition_id = config.espn_slug or config.key
    edition_id = f"{competition_id}:2026-27"
    rules_version = f"{competition_id}-2026-27-v1"
    prefix = f"{league_key}-demo"
    model_id = f"{competition_id}:independent-poisson:v3-demo"
    dixon_model_id = f"{competition_id}:dixon-coles:v3-demo"
    stack_model_id = f"{competition_id}:calibrated-stack:v3-demo"
    teams = (
        [
            ("bel:club-brugge", "Club Brugge"),
            ("bel:union-sg", "Union St.-Gilloise"),
            ("bel:anderlecht", "Anderlecht"),
            ("bel:genk", "Racing Genk"),
            ("bel:antwerp", "Antwerp"),
            ("bel:cercle-brugge", "Cercle Brugge KSV"),
            ("bel:gent", "KAA Gent"),
            ("bel:kortrijk", "KV Kortrijk"),
            ("bel:mechelen", "KV Mechelen"),
            ("bel:westerlo", "KVC Westerlo"),
            ("bel:lommel", "Lommel SK"),
            ("bel:oh-leuven", "OH Leuven"),
            ("bel:raal-la-louviere", "RAAL La Louvière"),
            ("bel:charleroi", "Royal Charleroi SC"),
            ("bel:st-truiden", "Sint-Truidense"),
            ("bel:standard-liege", "Standard Liege"),
            ("bel:beveren", "Waasland-Beveren"),
            ("bel:waregem", "Zulte-Waregem"),
        ]
        if league_key == "belgium"
        else [
            (f"{league_key}:arsenal", "Arsenal"),
            (f"{league_key}:liverpool", "Liverpool"),
            (f"{league_key}:man-city", "Manchester City"),
            (f"{league_key}:chelsea", "Chelsea"),
            (f"{league_key}:aston-villa", "Aston Villa"),
            (f"{league_key}:bournemouth", "AFC Bournemouth"),
            (f"{league_key}:brentford", "Brentford"),
            (f"{league_key}:brighton", "Brighton & Hove Albion"),
            (f"{league_key}:burnley", "Burnley"),
            (f"{league_key}:crystal-palace", "Crystal Palace"),
            (f"{league_key}:everton", "Everton"),
            (f"{league_key}:fulham", "Fulham"),
            (f"{league_key}:leeds", "Leeds United"),
            (f"{league_key}:man-united", "Manchester United"),
            (f"{league_key}:newcastle", "Newcastle United"),
            (f"{league_key}:nottingham-forest", "Nottingham Forest"),
            (f"{league_key}:sunderland", "Sunderland"),
            (f"{league_key}:tottenham", "Tottenham Hotspur"),
            (f"{league_key}:west-ham", "West Ham United"),
            (f"{league_key}:wolves", "Wolverhampton Wanderers"),
        ]
    )
    if len(teams) != config.team_count:
        raise RuntimeError(
            f"Demo registry has {len(teams)} teams; {league_key} requires {config.team_count}"
        )
    fixtures, forecasts, matrix_ids, matrices, explanations = [], [], [], [], []
    rates = [(1.65, 1.05), (1.30, 1.28), (1.18, 1.42)]
    pairings = [(0, 1), (2, 3), (1, 2)]
    for index, ((home_index, away_index), (home_rate, away_rate)) in enumerate(
        zip(pairings, rates), start=1
    ):
        fixture_id = f"{prefix}-{index}"
        kickoff = generated + timedelta(days=index, hours=15)
        home_id, home_name = teams[home_index]
        away_id, away_name = teams[away_index]
        grid = independent_poisson_grid(home_rate, away_rate)
        matrix = grid.normalized_mass()
        markets = markets_from_score_matrix(matrix)
        fixtures.append({
            "fixture_id": fixture_id,
            "edition_id": edition_id,
            "kickoff_utc": kickoff,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_display_name": home_name,
            "away_display_name": away_name,
            "status": "scheduled",
            "venue_name": f"{home_name} Stadium",
            "provider_event_id": f"demo-event-{index}",
            "source": "demo",
            "observed_at": generated,
            "rules_version": rules_version,
        })
        p_home, p_draw, p_away = (
            float(markets["p_home"]), float(markets["p_draw"]), float(markets["p_away"])
        )
        uncertainty = 0.06
        forecasts.append({
            "fixture_id": fixture_id,
            "issued_at": generated,
            "model_id": model_id,
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "p_home_lower80": max(0, p_home - uncertainty),
            "p_home_upper80": min(1, p_home + uncertainty),
            "p_home_lower50": max(0, p_home - uncertainty * 0.55),
            "p_home_upper50": min(1, p_home + uncertainty * 0.55),
            "p_draw_lower80": max(0, p_draw - 0.04),
            "p_draw_upper80": min(1, p_draw + 0.04),
            "p_draw_lower50": max(0, p_draw - 0.022),
            "p_draw_upper50": min(1, p_draw + 0.022),
            "p_away_lower80": max(0, p_away - uncertainty),
            "p_away_upper80": min(1, p_away + uncertainty),
            "p_away_lower50": max(0, p_away - uncertainty * 0.55),
            "p_away_upper50": min(1, p_away + uncertainty * 0.55),
            "leader_stability": 0.82 - index * 0.06,
            "cold_start": "full" if index < 3 else "partial_history",
            "cold_start_label": "Partial history" if index == 3 else "Full history",
            "home_history_n": 74 - index,
            "away_history_n": 65 - index,
            "effective_sample_size": 65 - index,
            "cold_start_status": "full_history" if index < 3 else "partial_history",
            "prior_weight": 0.05,
            "entity_resolution_status": "provider_alias",
            "expected_total_goals": home_rate + away_rate,
            "p_btts_yes": float(markets["p_btts_yes"]),
            "home_strength_probability": p_home - 0.03,
            "away_strength_probability": p_away + 0.03,
            "calibrated_p_draw": 0.97 * p_draw + 0.03 / 3,
            "score_entropy": float(-(matrix[matrix > 0] * np.log(matrix[matrix > 0])).sum()),
            "shots_tempo": 20.0 + index * 1.4,
            "style_matchup": 0.42 + index * 0.12,
        })
        matrix_ids.append(fixture_id)
        matrices.append(matrix)
        for outcome, contribution, display, explanation in (
            ("home", 0.08, "Home attacking strength", f"{home_name}'s recent attack raises the home forecast."),
            ("home", -0.03, "Opponent away resistance", f"{away_name}'s away defense offsets part of that edge."),
            ("draw", 0.04, "Team-strength parity", "The teams' ratings keep the draw probability material."),
            ("away", 0.05, "Away transition threat", f"{away_name}'s scoring rate supports the away tail."),
        ):
            explanations.append({
                "fixture_id": fixture_id,
                "outcome": outcome,
                "display_name": display,
                "definition": display,
                "value": contribution,
                "contribution": contribution,
                "sample_timestamp": generated,
                "source": "demo team ledger",
                "explanation": explanation,
                "model_id": model_id,
                "model_version": __version__,
            })
    completed_fixture_id = f"{prefix}-historical-1"
    fixtures.append({
        "fixture_id": completed_fixture_id,
        "edition_id": edition_id,
        "kickoff_utc": generated - timedelta(days=14),
        "home_team_id": teams[0][0], "away_team_id": teams[2][0],
        "home_display_name": teams[0][1], "away_display_name": teams[2][1],
        "home_goals": 2, "away_goals": 1, "status": "completed",
        "venue_name": f"{teams[0][1]} Stadium",
        "provider_event_id": "demo-event-historical-1", "source": "demo",
        "observed_at": generated - timedelta(days=13), "rules_version": rules_version,
    })
    rows_by_artifact: dict[str, int | None] = {}
    for name, rows in (
        ("fixtures", fixtures),
        ("forecasts", forecasts),
        ("forecast_explanations", explanations),
    ):
        rows_by_artifact[name] = _frame(precomputed / f"{name}.parquet", rows)
    forecast_ledger = []
    fixture_by_id = {item["fixture_id"]: item for item in fixtures}
    for forecast in forecasts:
        fixture = fixture_by_id[forecast["fixture_id"]]
        base = np.array([
            forecast["p_home"], forecast["p_draw"], forecast["p_away"]
        ])
        for hours, label, blend in (
            (168, "initial", 0.10),
            (24, "24_hour", 0.04),
            (2, "lineup", 0.015),
            (0.1, "closing", 0.0),
        ):
            probability = (1 - blend) * base + blend * np.full(3, 1 / 3)
            forecast_ledger.append({
                "fixture_id": forecast["fixture_id"],
                "kickoff_utc": fixture["kickoff_utc"],
                "issued_at": fixture["kickoff_utc"] - timedelta(hours=hours),
                "model_id": model_id,
                "revision_label": label,
                "p_home": probability[0], "p_draw": probability[1],
                "p_away": probability[2],
                "effective_sample_size": min(
                    forecast["home_history_n"], forecast["away_history_n"]
                ),
                "cold_start_status": forecast["cold_start"],
                "result_status": "pending",
                "actual_outcome": None, "actual_home_goals": None,
                "actual_away_goals": None, "probability_log_loss": None,
                "probability_brier": None, "closing_line_value": None,
                "closing_comparison_status": "capability_unavailable",
            })
    historical_probability = np.array([0.52, 0.27, 0.21])
    forecast_ledger.append({
        "fixture_id": completed_fixture_id,
        "kickoff_utc": generated - timedelta(days=14),
        "issued_at": generated - timedelta(days=15),
        "model_id": model_id, "revision_label": "24_hour",
        "p_home": historical_probability[0], "p_draw": historical_probability[1],
        "p_away": historical_probability[2], "effective_sample_size": 61,
        "cold_start_status": "full_history", "result_status": "final",
        "actual_outcome": 0, "actual_home_goals": 2, "actual_away_goals": 1,
        "probability_log_loss": float(-np.log(historical_probability[0])),
        "probability_brier": float(
            np.square(historical_probability - np.array([1.0, 0.0, 0.0])).sum()
        ),
        "closing_line_value": None,
        "closing_comparison_status": "capability_unavailable",
    })
    rows_by_artifact["forecast_ledger"] = _frame(
        precomputed / "forecast_ledger.parquet", forecast_ledger
    )
    np.savez_compressed(
        precomputed / "score_matrices.npz",
        fixture_ids=np.asarray(matrix_ids),
        matrices=np.asarray(matrices, dtype=object),
    )
    # Re-write equal-size matrices without object dtype for safe allow_pickle=False reads.
    maximum = max(matrix.shape[0] for matrix in matrices)
    padded = np.zeros((len(matrices), maximum, maximum), dtype=float)
    for index, matrix in enumerate(matrices):
        padded[index, : matrix.shape[0], : matrix.shape[1]] = matrix
    np.savez_compressed(
        precomputed / "score_matrices.npz",
        fixture_ids=np.asarray(matrix_ids),
        matrices=padded,
    )
    rows_by_artifact["score_matrices"] = len(matrices)
    radars = build_fixture_radars(pd.DataFrame(forecasts)).merge(
        pd.DataFrame(fixtures)[[
            "fixture_id", "home_display_name", "away_display_name"
        ]], on="fixture_id"
    )
    radars["display_name"] = radars.home_display_name + " vs " + radars.away_display_name
    radars.to_parquet(precomputed / "radars.parquet", index=False)
    rows_by_artifact["radars"] = len(radars)
    snapshot_rows = []
    for index, (team_id, team_name) in enumerate(teams, start=1):
        snapshot_rows.append({
            "team_id": team_id, "team_name": team_name, "power_rank": index,
            "elo_rating": 1580 - index * 25, "matches": 20,
            "points_l5": 12 - (index % 5),
            "points_per_match_l10": 1.4 + index * 0.08, "attack_l10": 1.3 + index * 0.06,
            "defense_l10": 1.2 - index * 0.04, "goal_difference_l10": 0.2 + index * 0.05,
            "clean_sheet_rate_l10": 0.2 + index * 0.03, "home_goals_for_shrunk": 1.4,
            "away_goals_for_shrunk": 1.2, "home_sample": 10, "away_sample": 10,
            "xg_for_ewm10": 1.25 + index * 0.04,
            "xg_against_ewm10": 1.18 - index * 0.03,
            "shots_for_ewm10": 11.0 + index * 0.7,
            "shot_quality_ewm10": 0.105 + index * 0.004,
            "finishing_vs_expectation_ewm10": -0.08 + index * 0.05,
            "opponent_adjusted_points_l10": 1.36 + index * 0.07,
            "projection_expected_points": 64 - index * 2,
            "projection_expected_position": float(index),
            "rating_change_week": 3.0 - index * 0.6,
            "discipline_rate_l10": 1.4 + index * 0.12,
            "recovery_load": 0.34 + index * 0.05,
        })
    rows_by_artifact["team_snapshots"] = _frame(
        precomputed / "team_snapshots.parquet", snapshot_rows
    )
    standings_rows = []
    for index, (team_id, team_name) in enumerate(teams, start=1):
        wins = max(2, 14 - (index + 1) // 2)
        draws = 3 + index % 4
        losses = 20 - wins - draws
        goals_for = max(14, 44 - index)
        goals_against = 17 + index
        standings_rows.append({
            "position": index, "team_id": team_id, "team_name": team_name,
            "played": 20, "wins": wins, "draws": draws, "losses": losses,
            "goals_for": goals_for, "goals_against": goals_against,
            "goal_difference": goals_for - goals_against,
            "points": wins * 3 + draws, "games_in_hand": index % 2,
            "phase": "regular", "rules_version": rules_version,
        })
    rows_by_artifact["standings"] = _frame(
        precomputed / "standings.parquet", standings_rows
    )
    team_events = []
    for team_id, team_name in teams:
        for offset in range(10):
            team_events.append({
                "fixture_id": f"history-{team_id}-{offset}", "team_id": team_id,
                "team_name": team_name, "opponent_id": teams[(offset + 1) % len(teams)][0],
                "opponent_name": teams[(offset + 1) % len(teams)][1],
                "kickoff_utc": generated - timedelta(days=7 * (10 - offset)),
                "venue_role": "home" if offset % 2 == 0 else "away",
                "goals_for": offset % 3, "goals_against": (offset + 1) % 3,
                "points": [3, 1, 0][offset % 3], "result": ["W", "D", "L"][offset % 3],
                "score": f"{offset % 3}–{(offset + 1) % 3}",
                "xg_for": 0.8 + (offset % 4) * 0.3,
                "xg_against": 0.7 + ((offset + 1) % 4) * 0.25,
                "shots_for": 8 + offset % 7,
                "shot_quality": 0.09 + (offset % 5) * 0.01,
                "finishing_vs_expectation": (offset % 3) - (0.8 + (offset % 4) * 0.3),
                "opponent_adjusted_points": [2.8, 1.1, 0.2][offset % 3],
            })
    rows_by_artifact["team_events"] = _frame(precomputed / "team_events.parquet", team_events)
    rating_history = []
    for team_index, (team_id, team_name) in enumerate(teams):
        for week in range(8):
            rating_history.append({
                "fixture_id": f"rating-{team_id}-{week}",
                "team_id": team_id, "team_name": team_name,
                "kickoff_utc": generated - timedelta(days=7 * (8 - week)),
                "pre_match_rating": 1500 + team_index * 18 + week * 3,
                "post_match_rating": 1500 + team_index * 18 + (week + 1) * 3,
                "rating_deviation": max(60, 180 - week * 10),
            })
    rows_by_artifact["rating_history"] = _frame(
        precomputed / "rating_history.parquet", rating_history
    )
    style_rows = [{
        "team_id": team_id, "team_name": team_name,
        "style_label": ["high-event", "possession-control proxy", "balanced", "low-block proxy"][index % 4],
        "tempo_percentile": max(0.05, 0.95 - index / len(teams) * 0.85),
        "defense_percentile": 0.15 + ((index * 7) % len(teams)) / len(teams) * 0.75,
        "input_coverage": "aggregate fallback", "cluster_stability": 0.81,
        "definition": "League-relative aggregate style label; no tracking claim.",
    } for index, (team_id, team_name) in enumerate(teams)]
    rows_by_artifact["style_fingerprints"] = _frame(
        precomputed / "style_fingerprints.parquet", style_rows
    )
    difficulty_rows, recovery_rows = [], []
    for fixture in fixtures:
        for side, role in (("home", "home"), ("away", "away")):
            team_id = fixture[f"{side}_team_id"]
            opponent_id = fixture["away_team_id" if side == "home" else "home_team_id"]
            difficulty_rows.append({
                "fixture_id": fixture["fixture_id"], "team_id": team_id,
                "opponent_id": opponent_id, "kickoff_utc": fixture["kickoff_utc"],
                "venue_role": role, "difficulty_rating": 1480 + len(difficulty_rows) * 17,
                "difficulty_percentile": 0.35 + 0.08 * len(difficulty_rows),
                "expected_points": 1.35 if role == "home" else 1.05,
                "rating_observed_at": generated,
            })
            recovery_rows.append({
                "fixture_id": fixture["fixture_id"], "team_id": team_id,
                "kickoff_utc": fixture["kickoff_utc"], "days_since_previous": 6 + len(recovery_rows) % 3,
                "matches_previous_14d": 2, "short_rest": False,
                "travel_km": np.nan if role == "home" else 94.0 + 10 * len(recovery_rows),
                "travel_confidence": "venue_verified" if role == "away" else "not_applicable",
                "recovery_load": 0.40 + 0.04 * len(recovery_rows),
            })
    rows_by_artifact["fixture_difficulty"] = _frame(
        precomputed / "fixture_difficulty.parquet", difficulty_rows
    )
    rows_by_artifact["recovery_load"] = _frame(
        precomputed / "recovery_load.parquet", recovery_rows
    )
    projections = []
    for index, (team_id, team_name) in enumerate(teams, start=1):
        row = {
            "current_position": index, "team_id": team_id, "team_name": team_name,
            "current_points": 40 - index, "expected_points": 64 - index * 2,
            "expected_position": float(index),
        }
        for label, positions in config.outcome_labels.items():
            distance = min(abs(index - position) for position in positions)
            row[f"p_{label}"] = float(max(0.01, min(0.98, 0.82 - distance * 0.16)))
        projections.append(row)
    rows_by_artifact["season_simulations"] = _frame(
        precomputed / "season_simulations.parquet", projections
    )
    position_rows = []
    position_matrix = np.array([
        [
            np.exp(-abs(position - projection["current_position"]))
            for position in range(1, len(teams) + 1)
        ]
        for projection in projections
    ], dtype=float)
    for _ in range(100):
        position_matrix /= position_matrix.sum(axis=1, keepdims=True)
        position_matrix /= position_matrix.sum(axis=0, keepdims=True)
    for team_index, projection in enumerate(projections):
        for position, probability in enumerate(position_matrix[team_index], start=1):
            position_rows.append({
                "team_id": projection["team_id"], "team_name": projection["team_name"],
                "position": position, "probability": probability,
            })
    rows_by_artifact["position_probabilities"] = _frame(
        precomputed / "position_probabilities.parquet", position_rows
    )
    points_targets = [
        {"target": "title", "position_threshold": 1, "p10": 65.0, "median": 68.0, "p90": 72.0},
        {"target": "top_four", "position_threshold": min(4, len(teams)), "p10": 57.0, "median": 60.0, "p90": 64.0},
    ]
    rows_by_artifact["points_targets"] = _frame(
        precomputed / "points_targets.parquet", points_targets
    )
    match_stakes = [{
        "fixture_id": fixture["fixture_id"], "index": 0.08 + index * 0.04,
        "simulations": 10_000, "common_random_numbers": True,
        "supporting_metric": "maximum named-outcome probability delta",
    } for index, fixture in enumerate(fixtures)]
    rows_by_artifact["match_stakes"] = _frame(
        precomputed / "match_stakes.parquet", match_stakes
    )
    phase_scenarios = [{
        "scenario": "current projection", "phase_id": "regular",
        "pool": "all", "points_transition": "none" if league_key != "belgium" else "ceil(points / 2)",
        "bracket_path": "configured by competition rules", "rules_version": rules_version,
    }]
    rows_by_artifact["phase_scenarios"] = _frame(
        precomputed / "phase_scenarios.parquet", phase_scenarios
    )
    storylines = [
        {"storyline": "Highest-stakes fixture", "fixture_id": f"{prefix}-1", "metric": "stakes_index", "value": 0.18, "link_path": "/match-center?fixture=", "edition_id": edition_id, "rules_version": rules_version},
        {"storyline": "Largest model upset", "fixture_id": f"{prefix}-3", "metric": "upset_index", "value": 0.09, "link_path": "/match-center?fixture=", "edition_id": edition_id, "rules_version": rules_version},
        {"storyline": "Biggest power-rating move", "fixture_id": completed_fixture_id, "metric": "rating_move", "value": 12.4, "link_path": "/prediction-history?fixture=", "edition_id": edition_id, "rules_version": rules_version},
        {"storyline": "Sharpest form swing", "fixture_id": completed_fixture_id, "metric": "form_swing", "value": 0.31, "link_path": "/prediction-history?fixture=", "edition_id": edition_id, "rules_version": rules_version},
        {"storyline": "Biggest model surprise", "fixture_id": completed_fixture_id, "metric": "surprise_score", "value": 1.27, "link_path": "/prediction-history?fixture=", "edition_id": edition_id, "rules_version": rules_version},
    ]
    rows_by_artifact["storylines"] = _frame(precomputed / "storylines.parquet", storylines)
    trend_rows = []
    for index in range(24):
        trend_rows.append({
            "kickoff_utc": generated - timedelta(days=7 * (24 - index)),
            "rolling_total_goals": 2.45 + 0.02 * np.sin(index / 3),
            "rolling_home_win": 0.44 + 0.01 * np.cos(index / 4),
            "rolling_draw": 0.27 + 0.01 * np.sin(index / 5),
            "rolling_home_advantage_goals": 0.24 + 0.01 * np.cos(index / 5),
            "rolling_cards": 4.2 + 0.1 * np.sin(index / 2),
            "rolling_tempo": 22.0 + 0.3 * np.cos(index / 3),
            "rolling_market_error": 0.04 + 0.002 * np.sin(index / 4),
            "sample_n": min(50, 10 + index * 2), "rules_version": rules_version,
            "edition_id": edition_id, "window_matches": min(50, 10 + index * 2),
        })
    rows_by_artifact["league_trends"] = _frame(
        precomputed / "league_trends.parquet", trend_rows
    )
    rows_by_artifact["competitive_balance"] = _frame(
        precomputed / "competitive_balance.parquet", [{
            "edition_id": edition_id, "normalized_strength_dispersion": 0.055,
            "title_herfindahl": 0.31, "parity_index": 0.78,
            "promotion_relegation_churn": 0.16, "teams": config.team_count,
        }]
    )
    rows_by_artifact["cross_league"] = _frame(
        precomputed / "cross_league.parquet", [
            {"competition_id": competition_id, "edition_id": edition_id, "aligned_season": "2026-27", "definition_version": "league-comparison-v1", "goals_per_match": 2.61, "draw_rate": 0.27, "tempo": 22.4, "home_advantage_goals": 0.25, "parity_index": 0.78, "forecast_log_loss": 1.03, "calibration_error": 0.043},
            {"competition_id": "benchmark.simple", "edition_id": "benchmark.simple:2026-27", "aligned_season": "2026-27", "definition_version": "league-comparison-v1", "goals_per_match": 2.74, "draw_rate": 0.25, "tempo": 23.1, "home_advantage_goals": 0.28, "parity_index": 0.72, "forecast_log_loss": 1.05, "calibration_error": 0.051},
        ]
    )
    rows_by_artifact["provider_runs"] = _frame(
        precomputed / "provider_runs.parquet", [
            {"provider": "demo-fixtures", "started_at": generated - timedelta(minutes=4), "completed_at": generated - timedelta(minutes=3), "status": "succeeded", "rows": len(fixtures), "coverage": 1.0, "message": "Canonical fixtures loaded."},
            {"provider": "demo-model", "started_at": generated - timedelta(minutes=3), "completed_at": generated - timedelta(minutes=1), "status": "succeeded", "rows": len(forecasts), "coverage": 1.0, "message": "Forecast artifacts generated."},
        ]
    )
    evaluation = []
    calibration = []
    rng = np.random.default_rng(42)
    for index in range(60):
        probability = rng.dirichlet([4.2, 2.5, 3.1])
        actual = int(rng.choice(3, p=probability))
        over_probability = float(np.clip(0.40 + probability[0] * 0.20, 0.05, 0.95))
        btts_probability = float(np.clip(0.35 + probability[1] * 0.25, 0.05, 0.95))
        actual_over = int(rng.random() < over_probability)
        actual_btts = int(rng.random() < btts_probability)
        for candidate_id, candidate_probability in (
            (model_id, probability),
            (dixon_model_id, probability * 0.92 + np.full(3, 1 / 3) * 0.08),
            (stack_model_id, probability * 0.86 + np.full(3, 1 / 3) * 0.14),
        ):
            evaluation.append({
                "fixture_id": f"eval-{index}", "model_id": candidate_id,
                "kickoff_utc": generated - timedelta(days=60 - index),
                "issued_at": generated - timedelta(days=61 - index),
                "edition_id": edition_id, "phase": "regular",
                "fold_id": f"rolling-{index // 10 + 1}",
                "prediction_source": "out_of_fold",
                "coverage_status": "full", "cold_start_status": "full_history",
                "actual_outcome": actual,
                "p_home": candidate_probability[0],
                "p_draw": candidate_probability[1],
                "p_away": candidate_probability[2],
                "p_over_2_5": over_probability,
                "actual_over_2_5": actual_over,
                "p_btts_yes": btts_probability,
                "actual_btts_yes": actual_btts,
            })
    for outcome in ("home", "draw", "away"):
        for bucket in range(1, 6):
            forecast = bucket / 6
            calibration.append({
                "market": "1x2", "outcome": outcome, "forecast_mean": forecast,
                "observed_rate": min(1.0, max(0.0, forecast + (bucket - 3) * 0.015)),
                "count": 20 + bucket, "prediction_source": "out_of_fold",
            })
    for outcome in ("over_2_5", "btts_yes"):
        for bucket in range(1, 6):
            forecast = bucket / 6
            calibration.append({
                "market": "goals", "outcome": outcome, "forecast_mean": forecast,
                "observed_rate": min(1.0, max(0.0, forecast + (3 - bucket) * 0.01)),
                "count": 18 + bucket, "prediction_source": "out_of_fold",
            })
    rows_by_artifact["evaluation_predictions"] = _frame(
        precomputed / "evaluation_predictions.parquet", evaluation
    )
    rows_by_artifact["calibration"] = _frame(precomputed / "calibration.parquet", calibration)
    cohort_rows = []
    for cohort in (
        "rolling_90d", "rolling_180d", "season:first_five_matchdays",
        "history:full_history", "favorite:balanced", "schedule:normal_rest",
        "phase:regular", "coverage:full", "confidence_decile:4",
    ):
        cohort_rows.append({
            "cohort": cohort, "model_id": model_id, "n": 60,
            "status": "available", "log_loss": 1.03, "brier": 0.62,
            "rps": 0.19, "draw_recall": 0.21,
            "ece": 0.043, "edition_id": edition_id,
            "lower_95": 0.58, "upper_95": 0.66,
        })
    rows_by_artifact["cohort_metrics"] = _frame(
        precomputed / "cohort_metrics.parquet", cohort_rows
    )
    drift_rows = [
        {"category": category, "metric": metric, "value": value,
         "severity": severity, "evidence": evidence, "suggested_action": action}
        for category, metric, value, severity, evidence, action in (
            ("data", "feature_psi", 0.07, "stable", "No material input shift.", "Continue monitoring."),
            ("prediction", "home_probability_psi", 0.11, "watch", "Small distribution movement.", "Review after next matchweek."),
            ("coverage", "canonical_entity_coverage", 1.0, "stable", "All active fixture teams resolve.", "No action."),
            ("performance", "rolling_brier_delta", 0.006, "stable", "Within frozen tolerance.", "Continue shadow evaluation."),
        )
    ]
    rows_by_artifact["drift_report"] = _frame(
        precomputed / "drift_report.parquet", drift_rows
    )
    capabilities = (
        {"name": "weather", "status": "available", "source": "demo weather", "observed_at": generated.isoformat(), "coverage": 1.0},
        {"name": "squads", "status": "unavailable", "source": "none", "observed_at": None, "coverage": 0.0, "message": "No squad provider configured."},
        {"name": "odds", "status": "unavailable", "source": "none", "observed_at": None, "coverage": 0.0, "message": "Market Lab is hidden without timestamped prices."},
    )
    planned_artifacts = set(rows_by_artifact) | {
        "model_registry", "quality_report", "research_experiments",
        "research_metrics", "research_calibration",
    }
    resolved_features = manifest_feature_statuses(
        available_artifacts=planned_artifacts,
        capabilities={item["name"]: item["status"] for item in capabilities},
    )
    evaluation_frame = pd.DataFrame(evaluation)
    evaluation_scores = {
        candidate_id: score_panel(
            group.actual_outcome.to_numpy(dtype=int),
            group[["p_home", "p_draw", "p_away"]].to_numpy(),
        )
        for candidate_id, group in evaluation_frame.groupby("model_id")
    }
    reproduced_selection = min(
        evaluation_scores,
        key=lambda candidate_id: (
            evaluation_scores[candidate_id].log_loss,
            evaluation_scores[candidate_id].brier,
        ),
    )
    model_registry = {
        "schema_version": 1,
        "production_model_id": model_id,
        "models": [
            {"model_id": model_id, "family": "independent_poisson", "track": "independent", "status": "champion", "trained_through": generated.isoformat(), "feature_set_version": "team-ledger-v1", "entity_registry_version": f"{prefix}-entities-v1", "rules_version": rules_version, "evaluation_artifact": "evaluation_predictions", "model_card_artifact": "model_card", "reproduction_command": f"python -m scripts.build_v3_demo --league {league_key}"},
            {"model_id": dixon_model_id, "family": "dixon_coles", "track": "independent", "status": "challenger", "trained_through": generated.isoformat(), "feature_set_version": "team-ledger-v1", "entity_registry_version": f"{prefix}-entities-v1", "rules_version": rules_version, "evaluation_artifact": "evaluation_predictions", "model_card_artifact": "model_card", "reproduction_command": f"python -m scripts.build_v3_demo --league {league_key}"},
            {"model_id": stack_model_id, "family": "calibrated_stack", "track": "independent", "status": "challenger", "trained_through": generated.isoformat(), "feature_set_version": "team-ledger-v1", "entity_registry_version": f"{prefix}-entities-v1", "rules_version": rules_version, "evaluation_artifact": "evaluation_predictions", "model_card_artifact": "model_card", "reproduction_command": f"python -m scripts.build_v3_demo --league {league_key}"},
        ],
        "release_gate": {
            "status": "passed" if reproduced_selection == model_id else "failed",
            "selected_model_id": reproduced_selection,
            "baseline_model_id": dixon_model_id,
            "evaluation_window": "rolling-origin demonstration rows",
            "log_loss": evaluation_scores[model_id].log_loss,
            "brier": evaluation_scores[model_id].brier,
            "rps": evaluation_scores[model_id].rps,
            "ece": 0.043, "selection_reproduced": reproduced_selection == model_id,
            "interval_coverage_50": 0.52, "interval_coverage_80": 0.81,
            "challenger_promoted": False,
            "reason": "Persisted rolling-origin rows reproduce the production selection; challengers remain shadow models.",
        },
        "feature_statuses": {
            item["feature_id"]: item["status"] for item in resolved_features
        },
    }
    (precomputed / "model_registry.json").write_text(
        json.dumps(model_registry, indent=2) + "\n", encoding="utf-8"
    )
    rows_by_artifact["model_registry"] = 1
    (precomputed / "model_card.json").write_text(json.dumps({
        "title": f"{config.display_name} v3 demo model card",
        "production_model_id": model_id,
        "intended_use": "Chronological product demonstration; not a betting claim.",
        "reproduction_command": f"python -m scripts.build_v3_demo --league {league_key}",
    }, indent=2) + "\n", encoding="utf-8")
    rows_by_artifact["model_card"] = 1
    quality = {
        "publishable": True,
        "checks": [
            {"status": "passed", "severity": "blocking", "check": "entity_coverage", "observed": "4/4", "expected": "4/4", "message": "All active demo teams resolve."},
            {"status": "passed", "severity": "blocking", "check": "utc_and_edition_coverage", "observed": f"{len(fixtures)}/{len(fixtures)}", "expected": f"{len(fixtures)}/{len(fixtures)}", "message": f"All fixtures carry UTC kickoff, {edition_id}, and {rules_version}."},
            {"status": "passed", "severity": "blocking", "check": "point_in_time_features", "observed": 0, "expected": 0, "message": "No persisted pre-match timestamp is at or after kickoff."},
            {"status": "passed", "severity": "blocking", "check": "forecast_metadata", "observed": "effective sample and cold-start present", "expected": "present", "message": "Every scheduled forecast exposes history coverage."},
            {"status": "passed", "severity": "blocking", "check": "probability_coherence", "observed": 0, "expected": 0, "message": "All score and 1X2 distributions reconcile."},
            {"status": "passed", "severity": "blocking", "check": "independent_track_separation", "observed": 0, "expected": 0, "message": "Independent model rows contain no odds-derived input."},
            {"status": "passed", "severity": "blocking", "check": "rule_aware_table", "observed": rules_version, "expected": rules_version, "message": "Standings and projection artifacts use the configured edition rules."},
        ],
    }
    require_quality_report_payload(quality)
    (precomputed / "quality_report.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )
    rows_by_artifact["quality_report"] = 1
    experiment_spec = ExperimentSpec(
        experiment_id=f"{prefix}-distributions-v1",
        hypothesis="Dependent score models must beat the frozen Poisson baseline.",
        created_at=generated,
        leagues=(competition_id,), tracks=("independent",),
        candidates=(model_id, dixon_model_id, stack_model_id),
        primary_metric="multiclass_log_loss",
        secondary_metrics=("multiclass_brier", "ranked_probability_score"),
        mandatory_cohorts=("draw", "early_season", "cold_start", "regular_phase"),
        tuning_window_end=datetime(2025, 6, 30, 23, 59, 59, tzinfo=timezone.utc),
        forward_test_start=datetime(2025, 7, 1, tzinfo=timezone.utc),
        forward_test_end=datetime(2026, 6, 30, 23, 59, 59, tzinfo=timezone.utc),
        paired_block_unit="matchweek", family_test="white_style_reality_check",
        promotion_minimum={
            "log_loss_relative_improvement": 0.005,
            "calibration_regression_allowed": False,
            "fallback_failure_rate_max": 0.001,
        },
    )
    experiments = [{
        "experiment_id": f"{prefix}-distributions-v1", "hypothesis": "Dependent score models must beat the frozen Poisson baseline.",
        "created_at": generated, "forward_test_range": "2025-07 through 2026-06",
        "family_test": "white_style_reality_check", "status": "active",
        "spec_hash": experiment_spec.spec_hash,
        "candidates": list(experiment_spec.candidates),
        "promotion_thresholds": dict(experiment_spec.promotion_minimum),
        "decision_markdown": "No challenger is promoted in this demonstration bundle.",
    }]
    metrics = [{
        "experiment_id": f"{prefix}-distributions-v1", "model_id": "independent-poisson",
        "track": "independent", "fixtures": 60, "log_loss": 1.03, "brier": 0.62,
        "rps": 0.19, "draw_reliability_gap": 0.03, "fit_failures": 0,
        "dispersion_residual": 0.04, "zero_residual": 0.01,
        "diagonal_residual": 0.02, "tail_residual": 0.003,
        "paired_delta": 0.0, "lower_95": -0.01, "upper_95": 0.01,
        "family_test_p_value": 1.0, "market_log_loss_delta": None,
        "fit_seconds": 0.18, "inference_ms": 0.3, "artifact_bytes": 18_000,
        "fallback_rate": 0.0, "status": "champion",
    }]
    research_calibration = [
        {"experiment_id": f"{prefix}-distributions-v1", **row} for row in calibration
    ]
    for name, rows in (
        ("research_experiments", experiments),
        ("research_metrics", metrics),
        ("research_calibration", research_calibration),
    ):
        rows_by_artifact[name] = _frame(precomputed / f"{name}.parquet", rows)
    write_experiment_log(
        precomputed / "experiment_log.json",
        experiment_spec,
        {
            model_id: {"status": "champion", "threshold_passed": True},
            dixon_model_id: {"status": "challenger", "threshold_passed": False},
            stack_model_id: {"status": "challenger", "threshold_passed": False},
        },
    )
    rows_by_artifact["experiment_log"] = 1
    descriptors = []
    dependencies = {
        "forecasts": ("fixtures",), "score_matrices": ("fixtures", "forecasts"),
        "forecast_explanations": ("fixtures", "forecasts"), "radars": ("forecasts",),
        "forecast_ledger": ("fixtures", "forecasts"),
        "team_events": ("fixtures",), "team_snapshots": ("team_events",),
        "standings": ("fixtures",),
        "rating_history": ("team_events",),
        "style_fingerprints": ("team_snapshots",),
        "fixture_difficulty": ("fixtures", "rating_history", "forecasts"),
        "recovery_load": ("fixtures",),
        "season_simulations": ("score_matrices",), "storylines": ("radars", "season_simulations"),
        "position_probabilities": ("season_simulations",),
        "points_targets": ("season_simulations",),
        "match_stakes": ("season_simulations", "score_matrices"),
        "phase_scenarios": ("season_simulations",),
        "league_trends": ("team_events",),
        "competitive_balance": ("rating_history", "season_simulations"),
        "evaluation_predictions": ("forecasts",), "calibration": ("evaluation_predictions",),
        "cohort_metrics": ("evaluation_predictions",),
        "drift_report": ("evaluation_predictions", "provider_runs"),
        "model_registry": ("evaluation_predictions", "calibration", "model_card"),
        "quality_report": ("fixtures", "forecasts", "score_matrices", "provider_runs"),
        "research_metrics": ("research_experiments",),
        "research_calibration": ("research_experiments", "research_metrics"),
        "experiment_log": ("research_experiments", "research_metrics"),
    }
    for name, rows in rows_by_artifact.items():
        suffix = ".npz" if name == "score_matrices" else ".json" if name in {"model_registry", "quality_report", "model_card", "experiment_log"} else ".parquet"
        media = "application/octet-stream" if suffix == ".npz" else "application/json" if suffix == ".json" else "application/vnd.apache.parquet"
        descriptors.append(descriptor_for_file(
            root=root, name=name, path=f"{release_relative}/{name}{suffix}", media_type=media,
            schema_name=name, schema_version=1, rows=rows, generated_at=generated,
            producer="scripts.build_v3_demo", dependencies=dependencies.get(name, ()),
            model_id=model_id if name in {"forecasts", "score_matrices", "evaluation_predictions"} else None,
            rules_version=rules_version,
        ))
    manifest = ManifestV3(
        league=league_key, edition_id=edition_id, core_version=__version__,
        rules_version=rules_version,
        entity_registry_version=f"{prefix}-entities-v1", generated_at=generated.isoformat(),
        artifacts=tuple(descriptors),
        capabilities=capabilities,
        feature_statuses=resolved_features,
    )
    manifest_path = publish_manifest(
        manifest, manifest_dir / "cache_manifest.json", root=root
    )
    (root / "app.py").write_text(
        "from pitch_oracle_core import get_league_config, run_app\n"
        f"run_app(get_league_config({league_key!r}), root={str(root)!r})\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output/v3-demo")
    parser.add_argument("--league", default="belgium")
    args = parser.parse_args()
    path = build(Path(args.root), league_key=args.league)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
