from datetime import datetime, timezone
import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pitch_oracle_core.artifacts.manifest import ManifestV3, descriptor_for_file, write_manifest, validate_artifact_files
from pitch_oracle_core.artifacts.repository import ArtifactRepository
from pitch_oracle_core.cache import CacheRequirement, write_cache_manifest, validate_cache
from pitch_oracle_core.pitchapi.artifacts import publish_index
from pitch_oracle_core.pitchapi.storage import write_frame
from pitch_oracle_core.ui.pitchapi_analytics import latest_revision


def test_optional_integrity_failure_preserves_required_cache(tmp_path):
    (tmp_path / "data_files").mkdir()
    (tmp_path / "required.txt").write_text("forecast")
    (tmp_path / "data_files/pitchapi_health.json").write_text('{"capabilities":{}}')
    write_cache_manifest(tmp_path, requirements=(CacheRequirement("forecast", "required.txt"),), league="epl")
    (tmp_path / "data_files/pitchapi_health.json").write_text("broken")
    warnings = validate_cache(tmp_path, expected_league="epl")
    assert "Optional artifact" in warnings[0]
    repository = ArtifactRepository.from_manifest(tmp_path, expected_league="epl")
    assert repository.available("forecast")
    assert not repository.available("pitchapi_health")
    assert "pitchapi_health" in repository.manifest["optional_artifact_failures"]
    (tmp_path / "required.txt").write_text("bad")
    with pytest.raises(RuntimeError, match="integrity"):
        validate_cache(tmp_path, expected_league="epl")


def test_optional_v3_dependency_failures_do_not_serve_corrupt_data(tmp_path):
    generated = datetime.now(timezone.utc)
    descriptors = []
    for name, required, dependencies in (("forecast", True, ()), ("shots", False, ()), ("summary", False, ("shots",))):
        (tmp_path / f"{name}.txt").write_text(name)
        descriptors.append(descriptor_for_file(root=tmp_path, name=name, path=f"{name}.txt", media_type="text/plain", schema_name=name, schema_version=1, rows=1, generated_at=generated, producer="test", required=required, dependencies=dependencies))
    manifest = ManifestV3("epl", "2026", "test", "test", generated.isoformat(), tuple(descriptors))
    path = tmp_path / "precomputed/cache_manifest.json"
    write_manifest(manifest, path)
    (tmp_path / "shots.txt").unlink()
    with pytest.raises(FileNotFoundError):
        validate_artifact_files(manifest, tmp_path)
    repository = ArtifactRepository.from_manifest(tmp_path, expected_league="epl")
    assert repository.available("forecast")
    assert not repository.available("shots") and not repository.available("summary")


def test_latest_revision_excludes_removed_rows_and_uses_team_specific_clock():
    rows = pd.DataFrame({"team": ["one", "one", "two"], "player": ["removed", "current", "known"], "observed_at": ["2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z", "2026-08-01T00:00:00Z"]})
    assert latest_revision(rows, ["team"]).player.tolist() == ["current", "known"]


def _bundle(root):
    data = root / "data_files"
    data.mkdir()
    fixture = {"fixture_id": "fx:test", "kickoff_utc": "2026-08-01T18:00:00Z", "home_team_id": "epl:home", "away_team_id": "epl:away", "home_team": "Home", "away_team": "Away", "status": "finished", "observed_at": "2026-08-02T00:00:00Z"}
    write_frame(pd.DataFrame([fixture]), data / "pitchapi_matches.csv")
    shared = {"fixture_id": "fx:test", "observed_at": "2026-08-02T00:00:00Z"}
    write_frame(pd.DataFrame([{**shared, "team_id": team, "ppda": 8 + i, "field_tilt": 50 + i, "xt_total": .5 + i, "direct_speed": 2 + i, "vaep_offensive": .2, "vaep_defensive": .1} for i, team in enumerate(["epl:home", "epl:away"])]), data / "pitchapi_advanced_team.parquet")
    write_frame(pd.DataFrame([{**shared, "team_id": "epl:home", "player_id": "p:one", "player_name": "Player", "position": "MF", "minutes": 90, "rating": 7, "passes": 45}]), data / "pitchapi_player_match.parquet")
    write_frame(pd.DataFrame([{**shared, "team_id": "epl:home", "player_name": "Player", "minute": 30, "x": 90, "y": 34, "expected_goals": .2, "expected_goals_on_target": None, "is_goal": None}]), data / "pitchapi_shots.parquet")
    write_frame(pd.DataFrame([{**shared, "home_xg": .2, "away_xg": 0}]), data / "pitchapi_match_shot_features.csv")
    write_frame(pd.DataFrame([{**shared, "minute": 1, "home": .3, "away": .1}]), data / "pitchapi_momentum.parquet")
    write_frame(pd.DataFrame([{**shared, "team_id": "epl:home", "kind": "node", "player_id": "p:one", "player_name": "Player", "avg_x": 50, "avg_y": 30, "passes": 45, "passes_received": 30}]), data / "pitchapi_network.parquet")
    write_frame(pd.DataFrame([{**shared, "team_id": "epl:home", "kind": "team", "player_id": None, "player_name": None, "cell_x": 1, "cell_y": 2, "actions": 4, "coordinate_frame": "acting_ltr", "grid_length": 16, "grid_width": 12}]), data / "pitchapi_heatmaps.parquet")
    write_frame(pd.DataFrame([{ "fixture_id": "fx:test", "team_id": "epl:home", "snapshot_at": "2026-08-01T17:00:00Z", "player_name": "Player", "position": None, "is_starter": True, "lineup_status": "confirmed", "formation": "4-3-3"}]), data / "pitchapi_lineup_snapshots.parquet")
    write_frame(pd.DataFrame([{ "fixture_id": "fx:test", "issued_at": "2026-08-01T17:00:00Z", "revision_label": "lineup", "p_home": .4, "p_draw": .3, "p_away": .3, "model_id": "baseline", "fallback_reason": "Missing family data"}]), data / "pitchapi_forecast_revisions.parquet")
    (data / "pitchapi_health.json").write_text(json.dumps({"checked_at": "2026-08-02T00:00:00Z", "capabilities": {"shots": {"status": "stale", "coverage": .5, "observed_at": "2026-08-02T00:00:00Z"}}}))
    publish_index(root, "epl")


@pytest.mark.parametrize("page", ["match", "team", "model"])
def test_optional_analytics_all_sections_and_empty_state(tmp_path, page, monkeypatch):
    monkeypatch.delenv("PITCH_ORACLE_DATA_DIR", raising=False)
    _bundle(tmp_path)
    entry = tmp_path / "app.py"
    entry.write_text("from pitch_oracle_core import get_league_config\nfrom pitch_oracle_core.ui.pitchapi_analytics import legacy_context, render_" + page + "_page\nrender_" + page + "_page(legacy_context(get_league_config('epl'), " + repr(str(tmp_path)) + "))\n")
    app = AppTest.from_file(entry, default_timeout=30).run()
    assert not app.exception
    sections = ["Lineups", "Forecast history", "Shots", "Momentum", "Team comparison", "Players", "Passing network", "Heatmaps"] if page == "match" else ["Trends", "Style radar", "Players"] if page == "team" else []
    for section in sections:
        app.segmented_control[0].set_value(section).run()
        assert not app.exception, f"{section}: {app.exception}"
    (tmp_path / "data_files/pitchapi_artifacts.json").unlink()
    app.run()
    assert not app.exception
    assert app.info
