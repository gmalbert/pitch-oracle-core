import json

from pitch_oracle_core.events.matchflow import (
    statsbomb_events,
    statsbomb_match_catalog,
    statsbomb_source_manifest,
)


def test_statsbomb_metadata_scopes_event_files(tmp_path):
    (tmp_path / "matches").mkdir()
    (tmp_path / "events").mkdir()
    (tmp_path / "matches" / "1.json").write_text(json.dumps([
        {
            "match_id": 42,
            "competition": {"competition_id": 11, "competition_name": "Test"},
            "season": {"season_id": 22, "season_name": "2024"},
            "home_team": {"home_team_name": "A"},
            "away_team": {"away_team_name": "B"},
        }
    ]), encoding="utf-8")
    (tmp_path / "events" / "42.json").write_text(json.dumps([
        {"match_id": 42, "type": {"name": "Shot"}},
        {"match_id": 42, "type": {"name": "Pass"}},
    ]), encoding="utf-8")
    catalog = statsbomb_match_catalog(tmp_path)
    assert catalog.iloc[0]["competition_id"] == 11
    events = statsbomb_events(tmp_path, competition_id=11).collect()
    assert len(events) == 2
    manifest = statsbomb_source_manifest(tmp_path)
    assert set(manifest["path"]) == {"events/42.json", "matches/1.json"}
