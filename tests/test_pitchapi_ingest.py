"""End-to-end mocked transport -> mapped, durable normalized artifacts."""

from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock
import pandas as pd
import pytest

from pitch_oracle_core.pitchapi.ingest import refresh_pitchapi
from pitch_oracle_core.pitchapi.client import PitchAPIError
from pitch_oracle_core.pitchapi.storage import append_revisions, read_frame
from pitch_oracle_core.fixtures.canonical import canonical_fixture_frame
from pitch_oracle_core.leagues import get_league_config

NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


def canonical():
    return pd.DataFrame([
        {"fixture_id": "fx:past", "home_display_name": "Ajax", "away_display_name": "PSV", "home_team_id": "nl:ajax", "away_team_id": "nl:psv", "kickoff_utc": "2026-09-30T18:00:00Z"},
        {"fixture_id": "fx:next", "home_display_name": "PSV", "away_display_name": "Ajax", "home_team_id": "nl:psv", "away_team_id": "nl:ajax", "kickoff_utc": "2026-10-02T18:00:00Z"},
    ])


def transport():
    client = Mock()
    client.league_matches.return_value = [
        {"id": "m_past", "date": "2026-09-30", "time_utc": "2026-09-30T18:00:00Z", "status": "finished", "home_team": {"id": "t_a", "name": "Ajax"}, "away_team": {"id": "t_p", "name": "PSV"}},
        {"id": "m_next", "date": "2026-10-02", "time_utc": "2026-10-02T18:00:00Z", "status": "not_started", "home_team": {"id": "t_p", "name": "PSV"}, "away_team": {"id": "t_a", "name": "Ajax"}},
    ]

    def request(path):
        if path.endswith('/shots'):
            return {"periods": [{"period": "FirstHalf", "shots": [{"id": "s_1", "team_id": "t_a", "player": {"id": "p_a", "name": "One"}, "expected_goals": .5, "is_on_target": False, "x": 90, "y": 34}]}]}
        if path.endswith('/advanced'):
            return {"teams": [{"team": {"id": "t_a", "name": "Ajax"}, "possession_value": {"xt_total": 1}, "passing": {"pass_accuracy": 80}}]}
        if path.endswith('/advanced/players'):
            return {"players": [{"player": {"id": "p_a", "name": "One"}, "team_id": "t_a", "minutes_played": 90, "possession_value": {"vaep_offensive": 1}}]}
        if path.endswith('/lineups'):
            return {"home": {"confirmed": False, "starters": [{"player_id": "p_a", "name": "One"}]}, "away": {"confirmed": True, "starters": [{"player_id": "p_b", "name": "Two"}]}}
        if path.endswith('/momentum'):
            return {"points": [{"minute": 10, "value": .2}]}
        if path.endswith('/advanced/network'):
            return {"networks": [{"team": {"id": "t_a"}, "nodes": [{"player": {"id": "p_a"}, "avg_x": 50, "avg_y": 34}], "edges": []}]}
        if path.endswith('/heatmaps'):
            return {"grid": {"frame": "acting_ltr", "length": 16, "width": 12}, "teams": [{"team": {"id": "t_a"}, "cells": [[5, 6, 2]]}]}
        raise AssertionError(path)
    client._request.side_effect = request
    return client


def run(tmp_path, client, **extra):
    return refresh_pitchapi("eredivisie", seasons=["2026/2027"], output_dir=tmp_path, canonical=canonical(), client=client, now=extra.pop("now", NOW), with_advanced=True, with_players=True, with_lineups=True, with_momentum=True, with_network=True, with_heatmaps=True, **extra)


def test_pipeline_maps_and_emits_every_requested_domain_artifact(tmp_path):
    frames = run(tmp_path, transport())
    for name in ("pitchapi_matches", "pitchapi_shots", "pitchapi_advanced_team", "pitchapi_player_match", "pitchapi_lineup_snapshots", "pitchapi_momentum", "pitchapi_network", "pitchapi_heatmaps", "pitchapi_match_shot_features", "pitchapi_match_xg"):
        assert not frames[name].empty, name
    assert set(frames["pitchapi_matches"].fixture_id) == {"fx:past", "fx:next"}
    assert frames["pitchapi_match_xg"].iloc[0].home_xg == .5
    assert read_frame(tmp_path / "pitchapi_shots.parquet").iloc[0].team_id == "nl:ajax"


def test_repeat_refresh_is_idempotent_and_does_not_refetch_heavy_payloads(tmp_path):
    client = transport()
    first = run(tmp_path, client)
    requests = client._request.call_count
    second = run(tmp_path, client)
    assert client._request.call_count == requests
    for name in first:
        pd.testing.assert_frame_equal(first[name].reset_index(drop=True), second[name].reset_index(drop=True), check_dtype=False)


def test_hourly_lineup_observations_are_preserved_even_if_xi_is_unchanged(tmp_path):
    client = transport()
    first = run(tmp_path, client)
    second = run(tmp_path, client, now=NOW + timedelta(hours=1))
    assert len(second["pitchapi_lineup_snapshots"]) == 2 * len(first["pitchapi_lineup_snapshots"])
    assert second["pitchapi_lineup_snapshots"].snapshot_at.nunique() == 2


def test_optional_refresh_failure_preserves_last_valid_cache(tmp_path):
    client = transport()
    first = run(tmp_path, client)
    client.league_matches.side_effect = PitchAPIError("INTERNAL_SERVER_ERROR")
    second = run(tmp_path, client, now=NOW + timedelta(days=1))
    pd.testing.assert_frame_equal(first["pitchapi_shots"], second["pitchapi_shots"], check_dtype=False)
    health = json.loads((tmp_path / "pitchapi_provider_run.json").read_text())
    assert health["status"] == "degraded"
    assert health["errors"][-1]["code"] == "INTERNAL_SERVER_ERROR"


def test_missing_credential_does_not_erase_existing_artifacts(tmp_path, monkeypatch):
    first = run(tmp_path, transport())
    monkeypatch.delenv("PITCH_API_KEY", raising=False)
    second = refresh_pitchapi("eredivisie", output_dir=tmp_path, now=NOW)
    pd.testing.assert_frame_equal(first["pitchapi_shots"], second["pitchapi_shots"], check_dtype=False)
    assert json.loads((tmp_path / "pitchapi_provider_run.json").read_text())["status"] == "unavailable"


def test_normalized_revision_cannot_be_rewritten(tmp_path):
    path = tmp_path / "values.parquet"
    frame = pd.DataFrame([{"fixture_id": "fx:past", "observed_at": NOW.isoformat(), "value": 1}])
    append_revisions(frame, path, keys=["fixture_id", "observed_at"])
    with pytest.raises(ValueError, match="rewrite"):
        append_revisions(frame.assign(value=2), path, keys=["fixture_id", "observed_at"])
    assert read_frame(path).iloc[0].value == 1


def test_canonical_historical_identity_is_independent_of_row_order_and_added_rows():
    config = get_league_config("eredivisie")
    raw = pd.DataFrame([{"Date": "2026-09-30", "Time": "20:00", "HomeTeam": "Ajax", "AwayTeam": "PSV"}])
    first = canonical_fixture_frame(raw, config)
    expanded = canonical_fixture_frame(pd.concat([raw.assign(Date="2026-09-20"), raw]), config)
    assert first.iloc[0].fixture_id == expanded.iloc[1].fixture_id
    assert first.iloc[0].home_team_id == "eredivisie:ajax"


def test_latest_season_does_not_depend_on_catalogue_order(tmp_path):
    client = transport()
    client.leagues.return_value = [{"id": "l_4H43wr", "seasons": ["2021/2022", "2026/2027", "2025/2026"]}]
    refresh_pitchapi("eredivisie", output_dir=tmp_path, canonical=canonical(), client=client, now=NOW)
    client.league_matches.assert_called_once_with("l_4H43wr", "2026/2027", status="all")


def test_football_data_historical_clock_is_independent_of_weather_timezone():
    config = get_league_config("eredivisie")
    raw = pd.DataFrame([{"Date": "2022-08-05", "Time": "19:00", "HomeTeam": "Heerenveen", "AwayTeam": "Sparta Rotterdam"}])
    result = canonical_fixture_frame(raw, config, input_timezone=config.sources.historical_timezone)
    assert result.iloc[0].kickoff_utc == pd.Timestamp("2022-08-05T18:00:00Z")


def test_legacy_eastern_schedule_clock_preserves_provider_utc_kickoff():
    config = get_league_config("laliga")
    raw = pd.DataFrame([{"Date": "2026-08-15", "Time": "01:30 PM ET", "HomeTeam": "Deportivo Alavés", "AwayTeam": "Getafe CF", "Status": "SCHEDULED"}])
    result = canonical_fixture_frame(raw, config, input_timezone=config.sources.upcoming_timezone)
    assert result.iloc[0].kickoff_utc == pd.Timestamp("2026-08-15T17:30:00Z")
    assert result.iloc[0].status == "scheduled"


def test_reconciliation_normalizes_large_candidate_table_once(monkeypatch):
    import pitch_oracle_core.fixtures.provider_mapping as module
    count = 0
    original = module.normalized_name
    def observed(value):
        nonlocal count
        count += 1
        return original(value)
    monkeypatch.setattr(module, "normalized_name", observed)
    size = 150
    times = pd.date_range("2026-01-01", periods=size, freq="D", tz="UTC")
    canonical_frame = pd.DataFrame({"fixture_id": [f"fx:{i}" for i in range(size)], "home_display_name": "Ajax", "away_display_name": "PSV", "kickoff_utc": times})
    provider = pd.DataFrame({"match_id": [f"provider:{i}" for i in range(size)], "home_team": "Ajax", "away_team": "PSV", "kickoff_utc": times})
    mapped, audit = module.reconcile_fixtures(provider, canonical_frame, aliases={"Amsterdam Football Club": "Ajax"})
    assert len(mapped) == size
    assert audit.status.eq("mapped").all()
    assert count <= size * 4 + 4
