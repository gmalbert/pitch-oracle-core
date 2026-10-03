"""Real provider response shapes, canonical reconciliation, and null semantics."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from pitch_oracle_core.fixtures.provider_mapping import map_provider_fixture, reconcile_fixtures
from pitch_oracle_core.pitchapi.normalize import normalize_shots, normalize_team_data, normalize_player_data, normalize_lineups, normalize_heatmaps
from pitch_oracle_core.features.shot_profile import shot_profile_for_team, build_match_shot_features, chance_concentration

NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
TEAMS = {"t_home": "nl:ajax", "t_away": "nl:psv"}
LINEAGE = {"fixture_id": "fx:one", "match_id": "m_one", "team_ids": TEAMS, "observed_at": NOW}


def canonical():
    return pd.DataFrame([{"fixture_id": "fx:one", "home_display_name": "Ajax", "away_display_name": "PSV", "home_team_id": "nl:ajax", "away_team_id": "nl:psv", "kickoff_utc": "2026-10-01T14:00:00Z"}])


def provider(**changes):
    return pd.Series({"match_id": "m_one", "home_team": "Ajax Amsterdam", "away_team": "PSV", "kickoff_utc": "2026-10-01T14:00:00Z", **changes})


def shots():
    return {"periods": [{"period": "FirstHalf", "shots": [{"id": "s_one", "team_id": "t_home", "player": {"id": "p_one", "name": "One"}, "minute": 10, "x": 90, "y": 30, "expected_goals": 0.4, "is_on_target": False, "situation": "RegularPlay", "shot_type": "RightFoot"}]}]}


def test_alias_matching_preserves_internal_identity():
    match = map_provider_fixture(provider(), canonical(), aliases={"Ajax Amsterdam": "Ajax"})
    assert match.fixture_id == "fx:one"
    assert match.provider_match_id == "m_one"


@pytest.mark.parametrize("reason,rows", [("ambiguous", 2), ("unmatched", 0)])
def test_ambiguous_or_missing_fixture_is_never_selected(reason, rows):
    frame = canonical() if rows else canonical().iloc[:0]
    if rows == 2:
        frame = pd.concat([frame, frame.assign(fixture_id="fx:two")])
    mapped, audit = reconcile_fixtures(pd.DataFrame([provider()]), frame, aliases={"Ajax Amsterdam": "Ajax"}, mapped_at=NOW)
    assert mapped.empty
    assert audit.iloc[0].status == reason


def test_reversed_teams_are_reported_and_rejected():
    mapped, audit = reconcile_fixtures(pd.DataFrame([provider(home_team="PSV", away_team="Ajax")]), canonical(), aliases={}, mapped_at=NOW)
    assert mapped.empty
    assert audit.iloc[0].status == "reversed"


def test_two_provider_events_cannot_own_same_canonical_fixture():
    incoming = pd.DataFrame([provider(), provider(match_id="m_two")])
    mapped, audit = reconcile_fixtures(incoming, canonical(), aliases={"Ajax Amsterdam": "Ajax"}, mapped_at=NOW)
    assert mapped.empty
    assert set(audit.status) == {"ambiguous"}


def test_shots_have_lineage_scoped_id_and_null_xgot():
    result = normalize_shots(shots(), **LINEAGE)
    assert result.iloc[0].team_id == "nl:ajax"
    assert result.iloc[0].player_id == "pitchapi:p_one"
    assert result.iloc[0].shot_id == "s_one"
    assert result.iloc[0].coordinate_frame == "acting_ltr"
    assert pd.isna(result.iloc[0].expected_goals_on_target)


@pytest.mark.parametrize("change", [{"team_id": "unknown"}, {"expected_goals": -1}, {"x": 106}, {"y": -1}, {"id": None}])
def test_invalid_shot_is_rejected(change):
    payload = shots()
    payload["periods"][0]["shots"][0].update(change)
    with pytest.raises(ValueError):
        normalize_shots(payload, **LINEAGE)


def test_duplicate_shot_ids_within_fixture_are_rejected():
    payload = shots()
    payload["periods"][0]["shots"] *= 2
    with pytest.raises(ValueError, match="unique ID"):
        normalize_shots(payload, **LINEAGE)


def test_advanced_field_names_match_provider_and_preserve_unknown_values():
    payload = {"teams": [{"team": {"id": "t_home"}, "passing": {"pass_accuracy": 83.0}, "tempo": {"direct_speed": 1.4}, "defending": {"ppda": None}}]}
    result = normalize_team_data(payload, **LINEAGE, home_provider_id="t_home")
    assert result.iloc[0].pass_accuracy == 83.0
    assert result.iloc[0].direct_speed == 1.4
    assert pd.isna(result.iloc[0].ppda)


def test_player_match_has_canonical_identity_minutes_and_keeper_exposure():
    payload = {"players": [{"player": {"id": "p_keeper", "name": "Keeper"}, "team_id": "t_home", "minutes_played": 90, "goalkeeping": {"claims": 5, "claims_won": 4, "claim_rate": 80, "distributions": 20}}]}
    result = normalize_player_data(payload, **LINEAGE)
    assert result.iloc[0].minutes == 90
    assert result.iloc[0].position == "GK"
    assert result.iloc[0].keeper_claims == 5
    assert result.iloc[0].keeper_claim_rate == 80


def test_home_and_away_confirmation_are_independent_and_slots_are_not_roles():
    payload = {"home": {"confirmed": True, "starters": [{"player_id": "p_one", "name": "One", "position_id": 0}]}, "away": {"confirmed": False, "lineup_type": "lastStarting11", "bench": [{"player_id": "p_two", "name": "Two", "position_id": 0}]}}
    frame = normalize_lineups(payload, fixture_id="fx:one", match_id="m_one", home_team_id="nl:ajax", away_team_id="nl:psv", snapshot_at=NOW, kickoff_utc=NOW)
    assert frame.iloc[0].lineup_status == "confirmed"
    assert frame.iloc[1].lineup_status == "predicted"
    assert pd.isna(frame.iloc[0].position)
    assert frame.iloc[1].position == "GK"
    assert frame.iloc[0].snapshot_id != frame.iloc[1].snapshot_id


def test_heatmap_orientation_and_grid_bounds_are_explicit():
    payload = {"grid": {"length": 16, "width": 12, "frame": "home_ltr"}, "teams": [{"team": {"id": "t_home"}, "side": "home", "cells": [[15, 11, 3]]}]}
    frame = normalize_heatmaps(payload, **LINEAGE)
    assert frame.iloc[0].coordinate_frame == "home_ltr"
    payload["teams"][0]["cells"] = [[16, 0, 1]]
    with pytest.raises(ValueError, match="outside"):
        normalize_heatmaps(payload, **LINEAGE)


def test_absent_shots_are_unknown_but_successful_empty_payload_means_zero_shots():
    frame = normalize_shots({"periods": []}, **LINEAGE)
    absent = shot_profile_for_team(frame, available=False)
    assert all(pd.isna(value) for value in absent.values())
    empty = shot_profile_for_team(frame, available=True)
    assert empty["shots"] == 0
    assert empty["xg"] == 0
    assert pd.isna(empty["xg_per_shot"])


def test_missing_on_target_xgot_is_not_invented_zero():
    payload = shots()
    payload["periods"][0]["shots"][0]["is_on_target"] = True
    frame = normalize_shots(payload, **LINEAGE)
    assert pd.isna(shot_profile_for_team(frame)["xgot"])


def test_unknown_xg_and_situation_are_not_zero_or_open_play():
    payload = shots()
    payload["periods"][0]["shots"][0]["expected_goals"] = None
    payload["periods"][0]["shots"][0]["situation"] = None
    profile = shot_profile_for_team(normalize_shots(payload, **LINEAGE))
    assert pd.isna(profile["xg"])
    assert pd.isna(profile["open_play_xg_share"])
    assert profile["xgot"] == 0  # known off-target, not an unavailable measurement


def test_concentration_and_fixture_summary():
    assert chance_concentration(pd.Series([.6, .4, .3, .1])) == pytest.approx(1.3 / 1.4)
    fixture = canonical().assign(shots_available=True)
    summary = build_match_shot_features(normalize_shots(shots(), **LINEAGE), fixture)
    assert summary.iloc[0].home_xg == .4
    assert summary.iloc[0].away_shots == 0
    assert pd.isna(summary.iloc[0].away_xg_per_shot)
