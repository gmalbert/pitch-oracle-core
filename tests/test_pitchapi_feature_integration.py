from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitch_oracle_core.features import no_odds_feature_columns, is_prematch_feature
from pitch_oracle_core.features.families import FeatureFamilyConfig
from pitch_oracle_core.features.pitchapi_mart import attach_pitchapi_features
from pitch_oracle_core.fixtures.canonical import canonical_fixture_frame
from pitch_oracle_core.leagues import get_league_config
from pitch_oracle_core.pitchapi.normalize import normalize_lineups
from pitch_oracle_core.players.goalkeeper import build_keeper_matches
from prepare_model_data import prepare_historical_features


def fixtures():
    return canonical_fixture_frame(pd.DataFrame([
        {"Date": "2026-08-01", "Time": "15:00", "HomeTeam": "Ajax", "AwayTeam": "PSV", "FTR": "H"},
        {"Date": "2026-08-08", "Time": "15:00", "HomeTeam": "Ajax", "AwayTeam": "PSV", "FTR": "H"},
        {"Date": "2026-08-15", "Time": "15:00", "HomeTeam": "Ajax", "AwayTeam": "PSV"},
    ]), get_league_config("eredivisie"))


def observations(f):
    return pd.DataFrame([
        {"fixture_id": f.iloc[0].fixture_id, "team_id": f.iloc[0].home_team_id, "observed_at": "2026-08-01T18:00:00Z", "xt_total": 2.0, "ppda": 8.0},
        {"fixture_id": f.iloc[0].fixture_id, "team_id": f.iloc[0].away_team_id, "observed_at": "2026-08-01T18:00:00Z", "xt_total": 1.0, "ppda": 12.0},
        {"fixture_id": f.iloc[1].fixture_id, "team_id": f.iloc[1].home_team_id, "observed_at": "2026-08-09T18:00:00Z", "xt_total": 99.0, "ppda": 1.0},
    ])


def test_training_and_serving_share_actual_cutoff_and_ignore_late_corrections():
    f = fixtures()
    cutoff = pd.Timestamp("2026-08-08T12:00:00Z")
    target = f.iloc[[1]].assign(as_of=cutoff)
    o = observations(f)
    first = attach_pitchapi_features(target, f, advanced=o)
    live = attach_pitchapi_features(f.iloc[[1]], f, advanced=o, as_of=cutoff)
    pd.testing.assert_frame_equal(first.drop(columns="as_of"), live.drop(columns="as_of"))
    assert first.iloc[0].home_xt_for_ewm10 == 2
    assert first.iloc[0].home_xt_against_ewm10 == 1
    correction = o.iloc[[0]].assign(observed_at="2026-08-08T14:00:00Z", xt_total=999999)
    poisoned = attach_pitchapi_features(target, f, advanced=pd.concat([o, correction]))
    pd.testing.assert_frame_equal(first, poisoned)


def test_unversioned_xg_never_populates_historical_replay(tmp_path: Path):
    f = fixtures().iloc[:2].assign(FTHG=2, FTAG=1, FTR="H")
    f.to_csv(tmp_path / "history.csv", sep="\t", index=False)
    xg = f[["HomeTeam", "AwayTeam", "MatchDate"]].rename(columns={"MatchDate": "match_date"}).assign(home_xg=999, away_xg=99)
    xg.to_csv(tmp_path / "xg.csv", sep="\t", index=False)
    result = prepare_historical_features(league_key="eredivisie", source=tmp_path / "history.csv", destination=tmp_path / "features.csv", xg_source=tmp_path / "xg.csv")
    assert result.home_xg.eq(999).all()
    assert result.home_xg_for_ewm10.isna().all()
    assert "home_xg" not in no_odds_feature_columns(result, enabled_families=("xg",))
    assert "home_xg_for_ewm10" not in no_odds_feature_columns(result)


def test_raw_provider_and_metadata_columns_cannot_enter_models():
    frame = pd.DataFrame(columns=["EloDiff", "home_xg", "home_ppda", "xt_total", "provider_schema_version", "home_player_rating", "observed_at", "home_xt_for_ewm10", "home_keeper_strength", "home_lineup_attack_delta"])
    assert no_odds_feature_columns(frame) == ["EloDiff"]
    assert no_odds_feature_columns(frame, enabled_families=("advanced_team", "goalkeeper")) == ["EloDiff", "home_xt_for_ewm10", "home_keeper_strength"]
    assert not is_prematch_feature("home_player_rating")
    with pytest.raises(ValueError, match="Unknown feature"):
        no_odds_feature_columns(frame, enabled_families=("anything",))


def test_promotion_config_requires_league_and_evidence(tmp_path):
    assert FeatureFamilyConfig.load(tmp_path / "none.json", league_key="epl").enabled_families == ()
    with pytest.raises(ValueError, match="evidence"):
        FeatureFamilyConfig("epl", ("xg",))
    path = tmp_path / "promotion.json"
    path.write_text('{"league_key":"epl","version":1,"enabled_families":["xg"],"evidence_id":"reviewed-run"}')
    assert FeatureFamilyConfig.load(path, league_key="epl").enabled_families == ("xg",)
    with pytest.raises(ValueError, match="different league"):
        FeatureFamilyConfig.load(path, league_key="eredivisie")


def test_date_only_fixture_uses_start_of_day_conservative_cutoff():
    f = fixtures()
    f.loc[1, "kickoff_precision"] = "date_only"
    f.loc[1, "kickoff_lower_bound_utc"] = pd.Timestamp("2026-08-07T22:00:00Z")
    o = observations(f)
    correction = o.iloc[[0]].assign(observed_at="2026-08-08T08:00:00Z", xt_total=999)
    result = attach_pitchapi_features(f.iloc[[1]], f, advanced=pd.concat([o, correction]))
    assert result.iloc[0].home_xt_for_ewm10 == 2
    assert pd.Timestamp(result.iloc[0].as_of) < pd.Timestamp("2026-08-07T22:00:00Z")


def test_lineup_bridge_handles_one_confirmed_team_without_future_player_history():
    f = fixtures()
    players = pd.DataFrame([{"fixture_id": f.iloc[0].fixture_id, "team_id": f.iloc[0].home_team_id, "player_id": f"pitchapi:{i}", "position": "CM", "minutes": 90, "observed_at": "2026-08-01T18:00:00Z", "vaep_offensive": float(i), "vaep_defensive": 1.0} for i in range(11)])
    target = f.iloc[1]
    at = pd.Timestamp("2026-08-08T12:00:00Z")
    payload = {"home": {"confirmed": True, "starters": [{"id": str(i), "name": str(i)} for i in range(11)], "bench": []}}
    lineup = normalize_lineups(payload, fixture_id=target.fixture_id, match_id="provider", home_team_id=target.home_team_id, away_team_id=target.away_team_id, snapshot_at=at.to_pydatetime(), kickoff_utc=target.kickoff_utc.to_pydatetime())
    result = attach_pitchapi_features(f.iloc[[1]], f, players=players, lineups=lineup, as_of=at).iloc[0]
    assert result.home_lineup_status == "confirmed"
    assert result.away_lineup_status == "none"
    assert result.home_lineup_coverage > 0
    assert np.isnan(result.away_lineup_attack_delta)
    future = players.assign(fixture_id=f.iloc[2].fixture_id, observed_at="2026-08-16T18:00:00Z", vaep_offensive=99999)
    mutated = attach_pitchapi_features(f.iloc[[1]], f, players=pd.concat([players, future]), lineups=lineup, as_of=at).iloc[0]
    assert result.home_lineup_attack_delta == mutated.home_lineup_attack_delta


def test_keeper_shot_attribution_uses_opponent_non_own_goals_and_revisions():
    f = fixtures().iloc[[0]]
    fixture = f.iloc[0]
    players = pd.DataFrame([{"fixture_id": fixture.fixture_id, "team_id": fixture.home_team_id, "player_id": "pitchapi:gk", "position": "GK", "minutes": 90, "observed_at": "2026-08-01T18:00:00Z"}])
    shots = pd.DataFrame([{"fixture_id": fixture.fixture_id, "opponent_id": fixture.home_team_id, "period": "FirstHalf", "is_own_goal": False, "is_on_target": True, "is_goal": True, "expected_goals_on_target": .7, "observed_at": "2026-08-01T18:30:00Z"}, {"fixture_id": fixture.fixture_id, "opponent_id": fixture.home_team_id, "period": "FirstHalf", "is_own_goal": True, "is_on_target": True, "is_goal": True, "expected_goals_on_target": .9, "observed_at": "2026-08-01T18:30:00Z"}])
    result = build_keeper_matches(players, shots, f)
    assert len(result) == 1
    assert result.iloc[0].xgot_faced == .7
    assert result.iloc[0].goals_conceded_non_own_goal == 1
    assert result.iloc[0].observed_at == "2026-08-01T18:30:00+00:00"
    assert build_keeper_matches(players.assign(minutes=60), shots, f).empty
    assert np.isnan(build_keeper_matches(players, shots.assign(is_own_goal=None), f).iloc[0].xgot_faced)
