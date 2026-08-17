from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from pitch_oracle_core._version import __version__
from pitch_oracle_core import get_league_config
from pitch_oracle_core.artifacts.manifest import (
    ManifestV3,
    descriptor_for_file,
    write_manifest,
)
from pitch_oracle_core.artifacts.verify import verify
from pitch_oracle_core.artifacts.repository import ArtifactRepository
from scripts.build_v3_demo import build as build_demo


def _write_v3_bundle(root: Path) -> None:
    generated = datetime.now(timezone.utc)
    precomputed = root / "precomputed"
    precomputed.mkdir(parents=True)
    fixtures = pd.DataFrame([{
        "fixture_id": "fx:test",
        "edition_id": "bel.1:2026-27",
        "kickoff_utc": "2026-08-15T18:00:00Z",
        "home_team_id": "bel:home",
        "away_team_id": "bel:away",
        "home_display_name": "Home Club",
        "away_display_name": "Away Club",
        "status": "scheduled",
        "venue_name": "Test Stadium",
    }])
    forecasts = pd.DataFrame([{
        "fixture_id": "fx:test",
        "issued_at": "2026-08-12T18:00:00Z",
        "model_id": "test:poisson:v1",
        "p_home": 0.45,
        "p_draw": 0.30,
        "p_away": 0.25,
        "p_home_lower80": 0.38,
        "p_home_upper80": 0.52,
        "p_draw_lower80": 0.24,
        "p_draw_upper80": 0.35,
        "p_away_lower80": 0.19,
        "p_away_upper80": 0.31,
        "cold_start": "full",
        "leader_stability": 0.8,
    }])
    fixtures.to_parquet(precomputed / "fixtures.parquet", index=False)
    forecasts.to_parquet(precomputed / "forecasts.parquet", index=False)
    home = np.array([0.30, 0.36, 0.22, 0.09, 0.03])
    away = np.array([0.42, 0.38, 0.15, 0.04, 0.01])
    matrix = np.outer(home, away)
    matrix /= matrix.sum()
    np.savez_compressed(
        precomputed / "score_matrices.npz",
        fixture_ids=np.array(["fx:test"]),
        matrices=np.array([matrix]),
    )
    descriptors = []
    for name, filename, media, schema, rows, dependencies in (
        ("fixtures", "fixtures.parquet", "application/vnd.apache.parquet", "fixtures", 1, ()),
        ("forecasts", "forecasts.parquet", "application/vnd.apache.parquet", "forecasts", 1, ("fixtures",)),
        ("score_matrices", "score_matrices.npz", "application/octet-stream", "score_matrices", 1, ("fixtures", "forecasts")),
    ):
        descriptors.append(descriptor_for_file(
            root=root,
            name=name,
            path=f"precomputed/{filename}",
            media_type=media,
            schema_name=schema,
            schema_version=1,
            rows=rows,
            generated_at=generated,
            producer="test",
            dependencies=dependencies,
        ))
    manifest = ManifestV3(
        league="belgium",
        edition_id="bel.1:2026-27",
        core_version=__version__,
        entity_registry_version="test-v1",
        generated_at=generated.isoformat(),
        artifacts=tuple(descriptors),
        capabilities=({
            "name": "weather",
            "status": "unavailable",
            "source": "none",
            "observed_at": None,
            "coverage": 0.0,
            "message": "Provider not configured",
        },),
    )
    write_manifest(manifest, precomputed / "cache_manifest.json")


def test_v3_overview_starts_without_training_dependencies(tmp_path, monkeypatch):
    _write_v3_bundle(tmp_path)
    entrypoint = tmp_path / "app.py"
    entrypoint.write_text(
        "from pitch_oracle_core import get_league_config, run_app\n"
        f"run_app(get_league_config('belgium'), root={str(tmp_path)!r})\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(entrypoint, default_timeout=30).run()
    assert not app.exception
    assert [item.value for item in app.title] == ["Belgian Pro League intelligence"]
    assert "Data freshness" in [item.value for item in app.subheader]


def test_match_center_fixture_selection_and_optional_empty_states(tmp_path, monkeypatch):
    _write_v3_bundle(tmp_path)
    entrypoint = tmp_path / "match.py"
    entrypoint.write_text(
        "import streamlit as st\n"
        "from pitch_oracle_core import get_league_config\n"
        "from pitch_oracle_core.ui.app import build_context\n"
        "from pitch_oracle_core.ui.pages.match_center import render\n"
        f"render(build_context(get_league_config('belgium'), {str(tmp_path)!r}))\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(entrypoint, default_timeout=30).run()
    assert not app.exception
    assert app.selectbox[0].value.startswith("Home Club vs Away Club")
    assert app.query_params["fixture"] == ["fx:test"]
    assert [item.value for item in app.title] == ["Home Club vs Away Club"]
    assert "Scoreline probabilities" in [item.value for item in app.subheader]
    assert "Goal outlook" in [item.value for item in app.subheader]
    assert "Download forecast card" in [item.label for item in app.get("download_button")]


def test_complete_demo_page_surface_renders_without_exceptions(tmp_path, monkeypatch):
    root = tmp_path / "demo"
    build_demo(root)
    monkeypatch.chdir(tmp_path)
    pages = (
        "overview", "match_center", "radars", "prediction_history",
        "team_center", "comparison",
        "standings", "projections", "league_lab", "model_lab", "data_control",
        "research_lab", "market_lab",
    )
    for page in pages:
        entrypoint = tmp_path / f"page_{page}.py"
        entrypoint.write_text(
            "from pitch_oracle_core import get_league_config\n"
            "from pitch_oracle_core.ui.app import build_context\n"
            f"from pitch_oracle_core.ui.pages.{page} import render\n"
            f"render(build_context(get_league_config('belgium'), {str(root)!r}))\n",
            encoding="utf-8",
        )
        app = AppTest.from_file(entrypoint, default_timeout=30).run()
        assert not app.exception, f"{page}: {app.exception}"
    assert any("capability-gated" in item.value for item in app.info)


def test_p0_p1_bundle_is_live_for_belgium_and_simple_format_league(tmp_path, monkeypatch):
    required = {
        "F01", "F02", "F03", "F04", "F05", "F07", "F08", "F09", "F10",
        "F13", "F14", "F15", "F17", "F19", "F26", "F27", "F28", "F36",
        "F37", "F38", "F39", "F40", "F43", "F44", "F45", "F46", "F47",
        "F48",
    }
    for league_key in ("belgium", "epl"):
        root = tmp_path / league_key
        build_demo(root, league_key=league_key)
        verify(root, strict=True)
        payload = json.loads(
            (root / "precomputed" / "cache_manifest.json").read_text(encoding="utf-8")
        )
        statuses = {item["feature_id"]: item["status"] for item in payload["feature_statuses"]}
        assert {statuses[feature_id] for feature_id in required} == {"shipped"}
        projections = ArtifactRepository.from_manifest(root).frame("season_simulations")
        expected_outcome = "p_champions_playoff" if league_key == "belgium" else "p_champions_league"
        assert expected_outcome in projections
        assert len(projections) == get_league_config(league_key).team_count
        positions = ArtifactRepository.from_manifest(root).frame("position_probabilities")
        assert positions.position.nunique() == get_league_config(league_key).team_count
        np.testing.assert_allclose(
            positions.groupby("team_id").probability.sum().to_numpy(), 1.0, atol=1e-8
        )
        np.testing.assert_allclose(
            positions.groupby("position").probability.sum().to_numpy(), 1.0, atol=1e-8
        )

        monkeypatch.chdir(tmp_path)
        entrypoint = root / "app.py"
        app = AppTest.from_file(entrypoint, default_timeout=30).run()
        assert not app.exception, f"{league_key}: {app.exception}"


def test_repository_rolls_back_only_to_a_valid_same_edition_manifest(tmp_path):
    root = tmp_path / "rollback"
    build_demo(root)
    primary = root / "precomputed" / "cache_manifest.json"
    previous = root / "precomputed" / "cache_manifest.previous.json"
    payload = json.loads(primary.read_text(encoding="utf-8"))
    previous.write_text(json.dumps(payload), encoding="utf-8")
    payload["previous_manifest"] = "precomputed/cache_manifest.previous.json"
    payload["artifacts"][0]["sha256"] = "0" * 64
    primary.write_text(json.dumps(payload), encoding="utf-8")
    repository = ArtifactRepository.from_manifest(root, expected_league="belgium")
    assert repository.manifest["serving_fallback"]["manifest"].endswith(
        "cache_manifest.previous.json"
    )
    assert repository.available("fixtures")


def test_publication_retains_immutable_previous_release_graph(tmp_path):
    root = tmp_path / "immutable-releases"
    first_path = build_demo(root)
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second_path = build_demo(root)
    second = json.loads(second_path.read_text(encoding="utf-8"))
    previous_path = root / second["previous_manifest"]
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    first_paths = {item["path"] for item in first["artifacts"]}
    second_paths = {item["path"] for item in second["artifacts"]}
    assert first_paths.isdisjoint(second_paths)
    assert {item["path"] for item in previous["artifacts"]} == first_paths
    second["artifacts"][0]["sha256"] = "0" * 64
    second_path.write_text(json.dumps(second), encoding="utf-8")
    repository = ArtifactRepository.from_manifest(root, expected_league="belgium")
    assert repository.manifest["serving_fallback"]["manifest"] == second["previous_manifest"]


def test_overview_meets_cold_and_warm_cache_budgets(tmp_path, monkeypatch):
    root = tmp_path / "performance"
    build_demo(root)
    monkeypatch.chdir(tmp_path)
    started = perf_counter()
    app = AppTest.from_file(root / "app.py", default_timeout=30).run()
    cold_seconds = perf_counter() - started
    started = perf_counter()
    app.run()
    warm_seconds = perf_counter() - started
    assert not app.exception
    assert cold_seconds < 3.0
    assert warm_seconds < 1.5
