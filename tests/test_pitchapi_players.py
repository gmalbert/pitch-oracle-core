"""Actual player/lineup/keeper calculations and forecast-time replay."""

from datetime import datetime, timezone
import pandas as pd
import pytest

from pitch_oracle_core.players.lineup_snapshots import latest_eligible_lineup, lineup_status, lineup_continuity
from pitch_oracle_core.players.strength_estimator import player_strength_frame, estimate_player_strengths
from pitch_oracle_core.players.goalkeeper import keeper_state

NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


def snapshot(team, time, status, count=11):
    return [{"fixture_id": "fx:one", "team_id": team, "player_id": f"{team}:p{i}", "snapshot_at": time, "kickoff_utc": "2026-10-01T18:00:00Z", "snapshot_id": f"{team}:{time}", "lineup_status": status, "is_starter": True} for i in range(count)]


def test_lineup_selection_keeps_both_teams_with_different_observation_times():
    history = pd.DataFrame(snapshot("home", "2026-09-30T18:00:00Z", "predicted") + snapshot("away", "2026-10-01T10:00:00Z", "predicted") + snapshot("home", "2026-10-01T17:05:00Z", "confirmed"))
    early = latest_eligible_lineup(history, fixture_id="fx:one", as_of=pd.Timestamp("2026-10-01T16:00:00Z"))
    late = latest_eligible_lineup(history, fixture_id="fx:one", as_of=pd.Timestamp("2026-10-01T17:30:00Z"))
    assert len(early) == len(late) == 22
    assert set(early.lineup_status) == {"predicted"}
    assert lineup_status(late.loc[late.team_id == "home"]) == "confirmed"
    assert lineup_status(late.loc[late.team_id == "away"]) == "predicted"


def test_postkickoff_lineup_and_later_corrections_cannot_enter_old_replay():
    history = pd.DataFrame(snapshot("home", "2026-10-01T17:00:00Z", "confirmed") + snapshot("home", "2026-10-01T18:01:00Z", "confirmed"))
    selected = latest_eligible_lineup(history, fixture_id="fx:one", as_of=pd.Timestamp("2026-10-01T19:00:00Z"))
    assert selected.snapshot_at.max() == pd.Timestamp("2026-10-01T17:00:00Z")


def test_empty_and_partial_lineups_are_never_confirmed():
    history = pd.DataFrame(snapshot("home", "2026-10-01T17:00:00Z", "confirmed", count=10))
    selected = latest_eligible_lineup(history, fixture_id="fx:one", as_of=pd.Timestamp("2026-10-01T17:30:00Z"), require_complete=True)
    assert selected.empty
    assert lineup_status(selected) == "none"
    assert lineup_status(history) == "incomplete"


def test_provider_reversion_does_not_erase_a_known_official_xi():
    history = pd.DataFrame(snapshot("home", "2026-10-01T17:00:00Z", "confirmed") + snapshot("home", "2026-10-01T17:30:00Z", "predicted"))
    selected = latest_eligible_lineup(history, fixture_id="fx:one", as_of=pd.Timestamp("2026-10-01T17:45:00Z"))
    assert lineup_status(selected) == "confirmed"


def test_continuity_requires_two_complete_prior_xis():
    prior = {f"p{i}" for i in range(11)}
    current = prior - {"p0"} | {"p11"}
    assert lineup_continuity(current, prior) == pytest.approx(10 / 11)
    assert pd.isna(lineup_continuity(set(), prior))


def player_history():
    return pd.DataFrame([
        {"fixture_id": "fx:a", "player_id": "p:one", "team_id": "tm:one", "observed_at": "2026-09-20T20:00:00Z", "minutes": 90, "position": "ST", "vaep_offensive": 1},
        {"fixture_id": "fx:b", "player_id": "p:one", "team_id": "tm:two", "observed_at": "2026-09-27T20:00:00Z", "minutes": 10, "position": "ST", "vaep_offensive": 0},
        {"fixture_id": "fx:c", "player_id": "p:two", "team_id": "tm:one", "observed_at": "2026-09-20T20:00:00Z", "minutes": 90, "position": "ST", "vaep_offensive": 0},
    ])


def test_player_rates_are_minute_weighted_and_transfer_history_is_preserved():
    result = player_strength_frame(player_history(), estimated_at=NOW)
    one = result.loc[result.player_id == "p:one"].iloc[0]
    assert one.team_id == "tm:two"
    assert one.history_matches == 2
    assert one.attack_per_90 > 0
    assert one.effective_minutes < 100
    assert one.metric_coverage < 1


def test_future_player_value_does_not_change_past_strength_or_scaler():
    history = player_history()
    baseline = player_strength_frame(history, estimated_at=NOW)
    future = history.iloc[[0]].assign(fixture_id="fx:future", observed_at="2026-10-02T20:00:00Z", vaep_offensive=100000)
    modified = player_strength_frame(pd.concat([history, future]), estimated_at=NOW)
    pd.testing.assert_frame_equal(baseline, modified)


def test_player_strength_contract_and_zero_minutes():
    history = player_history()
    assert set(estimate_player_strengths(history, estimated_at=NOW)) == {"p:one", "p:two"}
    assert estimate_player_strengths(history.assign(minutes=0), estimated_at=NOW) == {}


def test_keeper_uses_non_own_goals_exposure_and_explicit_forecast_cutoff():
    matches = pd.DataFrame([
        {"fixture_id": "fx:a", "player_id": "p:gk", "observed_at": "2026-09-25T20:00:00Z", "minutes": 90, "xgot_faced": 2, "goals_conceded_non_own_goal": 1, "shots_on_target_faced": 5, "keeper_claims": 5, "keeper_claims_won": 4},
        {"fixture_id": "fx:b", "player_id": "p:gk", "observed_at": "2026-10-01T17:00:00Z", "minutes": 90, "xgot_faced": 1000, "goals_conceded_non_own_goal": 0, "shots_on_target_faced": 10, "keeper_claims": 3, "keeper_claims_won": 3},
    ])
    state = keeper_state(matches, target_kickoff=pd.Timestamp("2026-10-01T18:00:00Z"), as_of=pd.Timestamp(NOW)).iloc[0]
    assert state.keeper_strength == pytest.approx(90 / (90 + 1350))
    assert state.shots_on_target_faced == 5
    assert state.keeper_claim_rate == .8
    assert state.sample_status == "strong_prior"
