from pathlib import Path

import pytest

import scripts.bootstrap_consumer as bootstrap_module
from scripts.bootstrap_consumer import bootstrap_consumer, copy_local_environment


def test_bootstrap_creates_specialized_turnkey_consumer(tmp_path: Path, monkeypatch):
    synthetic_core = tmp_path / "core"
    synthetic_core.mkdir()
    (synthetic_core / ".env").write_text(
        "FD_API_KEY=synthetic-test-secret\n", encoding="utf-8"
    )
    monkeypatch.setattr(bootstrap_module, "CORE_ROOT", synthetic_core)
    target = tmp_path / "scotland-soccer"
    result = bootstrap_module.bootstrap_consumer("scotland", tmp_path)

    assert result == target.resolve()
    assert 'get_league_config("scotland")' in (target / "config.py").read_text()
    workflow = (target / ".github" / "workflows" / "artifact-pipeline.yml").read_text()
    assert "league_key: scotland" in workflow
    assert "Scottish Premiership artifact pipeline" in workflow
    entrypoint = (target / "predictions.py").read_text()
    assert "The shared Pitch Oracle package is not installed" in entrypoint
    assert "python -m pip install -r requirements.txt" in entrypoint
    bootstrap = (target / "scripts" / "bootstrap_local.py").read_text()
    assert 'environment["PITCH_ORACLE_LEAGUE"] = LEAGUE_CONFIG.key' in bootstrap
    assert 'load_dotenv(ROOT / ".env"' in bootstrap
    assert '"-m", "pitch_oracle_core.audit_cli"' in bootstrap
    assert '"-m", "precompute_database"' in bootstrap
    prediction_builder = (target / "scripts" / "precompute_predictions.py").read_text()
    assert "add_weather_features" in prediction_builder
    assert "weather_cache_" in prediction_builder
    assert (target / "scripts" / "verify_consumer.py").is_file()
    assert (target / "precomputed" / ".gitkeep").is_file()
    assert not tuple(target.rglob("__pycache__"))
    assert (target / ".env.example").is_file()
    assert (target / ".env").is_file()
    assert "synthetic-test-secret" in (target / ".env").read_text(encoding="utf-8")


def test_local_environment_copy_is_secret_safe(tmp_path: Path):
    source = tmp_path / "source.env"
    source.write_text("FD_API_KEY=synthetic-test-secret\n", encoding="utf-8")
    destination = tmp_path / "consumer"
    destination.mkdir()
    (destination / ".gitignore").write_text(".env\n", encoding="utf-8")

    assert copy_local_environment(destination, source)
    assert (destination / ".env").read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_local_environment_copy_refuses_a_trackable_secret(tmp_path: Path):
    source = tmp_path / "source.env"
    source.write_text("FD_API_KEY=synthetic-test-secret\n", encoding="utf-8")
    destination = tmp_path / "consumer"
    destination.mkdir()
    (destination / ".gitignore").write_text("*.log\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to copy secrets"):
        copy_local_environment(destination, source)


def test_bootstrap_refuses_to_overwrite_existing_path(tmp_path: Path):
    target = tmp_path / "netherlands-soccer"
    target.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        bootstrap_consumer("eredivisie", tmp_path)


def test_bootstrap_rejects_league_without_baseline_fixture_provider(tmp_path: Path):
    with pytest.raises(ValueError, match="not consumer-ready"):
        bootstrap_consumer("portugal", tmp_path)
