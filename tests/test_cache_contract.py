import json

from pitch_oracle_core.cache import CacheRequirement, validate_cache, write_cache_manifest


def test_cache_manifest_round_trip(tmp_path):
    requirements = (CacheRequirement("fixture", "data/fixture.txt"),)
    artifact = tmp_path / "data" / "fixture.txt"
    artifact.parent.mkdir()
    artifact.write_text("cached", encoding="utf-8")

    manifest = write_cache_manifest(tmp_path, requirements=requirements, league="test")
    validate_cache(tmp_path)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["league"] == "test"
    assert payload["artifacts"]["fixture"]["bytes"] == 6


def test_cache_manifest_detects_tampering(tmp_path):
    requirements = (CacheRequirement("fixture", "data/fixture.txt"),)
    artifact = tmp_path / "data" / "fixture.txt"
    artifact.parent.mkdir()
    artifact.write_text("cached", encoding="utf-8")
    write_cache_manifest(tmp_path, requirements=requirements)
    artifact.write_text("changed", encoding="utf-8")

    try:
        validate_cache(tmp_path)
    except RuntimeError as exc:
        assert "failed integrity validation" in str(exc)
    else:
        raise AssertionError("tampered cache was accepted")
