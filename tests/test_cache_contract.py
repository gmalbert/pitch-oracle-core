import json
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from pitch_oracle_core.cache import CacheRequirement, validate_cache, write_cache_manifest
from pitch_oracle_core.features import FEATURE_POLICY_VERSION


def test_cache_manifest_round_trip(tmp_path):
    requirements = (CacheRequirement("fixture", "data/fixture.txt"),)
    artifact = tmp_path / "data" / "fixture.txt"
    artifact.parent.mkdir()
    artifact.write_text("cached", encoding="utf-8")

    manifest = write_cache_manifest(tmp_path, requirements=requirements, league="test")
    validate_cache(tmp_path, expected_league="test")

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["core_version"] == "1.3.12"
    assert payload["feature_policy_version"] >= 2
    assert payload["league"] == "test"
    assert payload["artifacts"]["fixture"]["bytes"] == 6


def test_cache_manifest_detects_tampering(tmp_path):
    requirements = (CacheRequirement("fixture", "data/fixture.txt"),)
    artifact = tmp_path / "data" / "fixture.txt"
    artifact.parent.mkdir()
    artifact.write_text("cached", encoding="utf-8")
    write_cache_manifest(tmp_path, requirements=requirements, league="test")
    artifact.write_text("changed", encoding="utf-8")

    try:
        validate_cache(tmp_path, expected_league="test")
    except RuntimeError as exc:
        assert "failed integrity validation" in str(exc)
    else:
        raise AssertionError("tampered cache was accepted")


def test_stale_manifest_can_run_in_explicit_compatibility_mode(tmp_path):
    requirements = (CacheRequirement("fixture", "data/fixture.txt"),)
    artifact = tmp_path / "data" / "fixture.txt"
    artifact.parent.mkdir()
    artifact.write_text("cached", encoding="utf-8")
    manifest_path = write_cache_manifest(tmp_path, requirements=requirements, league="test")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    payload.pop("core_version")
    payload.pop("feature_policy_version")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    warnings = validate_cache(tmp_path, strict_contract=False, expected_league="test")
    assert warnings and "compatibility mode" in warnings[0]


def _prediction_requirements():
    return (
        CacheRequirement("preprocessed_data", "precomputed/preprocessed_data.pkl"),
        CacheRequirement("ensemble", "models/ensemble_model.pkl"),
        CacheRequirement("upcoming_predictions", "data_files/upcoming_predictions.csv"),
    )


def _write_prediction_contract_files(root, model_width=1):
    (root / "precomputed").mkdir()
    (root / "models").mkdir()
    (root / "data_files").mkdir()
    artifact = {"feature_contract": {
        "version": FEATURE_POLICY_VERSION,
        "feature_names": ["Form"],
        "imputation_values": {"Form": 0.0},
    }}
    with (root / "precomputed" / "preprocessed_data.pkl").open("wb") as stream:
        pickle.dump(artifact, stream)
    with (root / "models" / "ensemble_model.pkl").open("wb") as stream:
        pickle.dump(SimpleNamespace(n_features_in_=model_width), stream)
    pd.DataFrame(columns=[
        "HomeWin_Prob", "Draw_Prob", "AwayWin_Prob", "PredictedResult",
        "Risk_Score", "Confidence_Score", "Risk_Category", "Recommendation",
        "PredictionGeneratedAt",
    ]).to_csv(root / "data_files" / "upcoming_predictions.csv", index=False)


def test_manifest_validates_prediction_contract_semantics(tmp_path):
    _write_prediction_contract_files(tmp_path)
    write_cache_manifest(tmp_path, requirements=_prediction_requirements(), league="test")


def test_manifest_rejects_model_feature_width_mismatch(tmp_path):
    _write_prediction_contract_files(tmp_path, model_width=2)
    with pytest.raises(RuntimeError, match="Ensemble expects 2 features"):
        write_cache_manifest(tmp_path, requirements=_prediction_requirements(), league="test")


def test_manifest_rejects_artifacts_from_another_league(tmp_path):
    requirements = (CacheRequirement("fixture", "data/fixture.txt"),)
    artifact = tmp_path / "data" / "fixture.txt"
    artifact.parent.mkdir()
    artifact.write_text("cached", encoding="utf-8")
    write_cache_manifest(tmp_path, requirements=requirements, league="epl")

    with pytest.raises(RuntimeError, match="belongs to league 'epl'"):
        validate_cache(tmp_path, expected_league="eredivisie")


def test_manifest_requires_a_league_identity(tmp_path, monkeypatch):
    monkeypatch.delenv("PITCH_ORACLE_LEAGUE", raising=False)
    requirements = (CacheRequirement("fixture", "data/fixture.txt"),)
    artifact = tmp_path / "data" / "fixture.txt"
    artifact.parent.mkdir()
    artifact.write_text("cached", encoding="utf-8")

    with pytest.raises(ValueError, match="league key"):
        write_cache_manifest(tmp_path, requirements=requirements)
