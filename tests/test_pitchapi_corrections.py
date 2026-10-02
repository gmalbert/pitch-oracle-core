from datetime import timedelta

import pandas as pd

from pitch_oracle_core.pitchapi.ingest import refresh_pitchapi
from pitch_oracle_core.pitchapi.revisions import complete_responses
from pitch_oracle_core.players.goalkeeper import build_keeper_matches, keeper_state
from pitch_oracle_core.players.strength_estimator import player_strength_frame
from test_pitchapi_ingest import NOW, canonical, transport


def test_empty_correction_removes_current_rows_without_erasing_past_replay(tmp_path):
    client = transport()
    kwargs = dict(league="eredivisie", seasons=["2026/2027"], output_dir=tmp_path, canonical=canonical(), client=client, with_advanced=True, with_players=True)
    first = refresh_pitchapi(**kwargs, now=NOW)
    client._request.side_effect = lambda path: {"periods": []} if path.endswith("shots") else {"players": []} if path.endswith("players") else {"teams": []}
    changed = refresh_pitchapi(**kwargs, now=NOW + timedelta(days=1))
    assert not first["pitchapi_shots"].empty
    assert complete_responses(changed["pitchapi_shots"], changed["pitchapi_response_revisions"], artifact="pitchapi_shots").empty
    assert changed["pitchapi_shots"].equals(first["pitchapi_shots"])
    # A known empty shot response is zero; removed team/player values remain unknown.
    assert changed["pitchapi_match_shot_features"].sort_values("observed_at").iloc[-1].home_xg == 0
    assert complete_responses(changed["pitchapi_advanced_team"], changed["pitchapi_response_revisions"], artifact="pitchapi_advanced_team").empty
    past = player_strength_frame(changed["pitchapi_player_match"], estimated_at=NOW)
    current = player_strength_frame(changed["pitchapi_player_match"], estimated_at=NOW + timedelta(days=1))
    assert len(past) == 1 and current.empty


def test_keeper_empty_shot_correction_and_removed_keeper_are_cutoff_aware():
    first, corrected, removed = "2026-09-20T20:00:00Z", "2026-09-21T20:00:00Z", "2026-09-22T20:00:00Z"
    fixtures = pd.DataFrame([{"fixture_id": "f", "kickoff_utc": "2026-09-20T15:00:00Z", "home_team_id": "home", "away_team_id": "away"}])
    players = pd.DataFrame([{"fixture_id": "f", "player_id": "gk", "team_id": "home", "observed_at": first, "position": "GK", "minutes": 90}, {"fixture_id": "f", "player_id": "gk", "team_id": "home", "observed_at": removed, "position": None, "minutes": 0, "is_deleted": True}])
    shots = pd.DataFrame([{"fixture_id": "f", "observed_at": first, "opponent_id": "home", "period": "FirstHalf", "is_own_goal": False, "is_on_target": True, "is_goal": False, "expected_goals_on_target": .7}])
    revisions = pd.DataFrame([{"fixture_id": "f", "artifact": "pitchapi_shots", "observed_at": first, "row_count": 1}, {"fixture_id": "f", "artifact": "pitchapi_shots", "observed_at": corrected, "row_count": 0}])
    matches = build_keeper_matches(players, shots, fixtures, response_revisions=revisions)
    assert matches.iloc[0].xgot_faced == .7
    assert matches.loc[pd.to_datetime(matches.observed_at, utc=True).eq(pd.Timestamp(corrected))].iloc[0].xgot_faced == 0
    assert matches.loc[pd.to_datetime(matches.observed_at, utc=True).eq(pd.Timestamp(removed))].iloc[0].minutes == 0
    kickoff = pd.Timestamp("2026-09-30T15:00:00Z")
    before = keeper_state(matches, target_kickoff=kickoff, as_of=pd.Timestamp(first) + pd.Timedelta(hours=1))
    after = keeper_state(matches, target_kickoff=kickoff, as_of=pd.Timestamp(removed) + pd.Timedelta(hours=1))
    assert before.iloc[0].effective_minutes == 90
    assert after.iloc[0].effective_minutes == 0
