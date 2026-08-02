from pathlib import Path

import pytest

from scripts.bootstrap_consumer import bootstrap_consumer


def test_bootstrap_creates_specialized_turnkey_consumer(tmp_path: Path):
    target = tmp_path / "scotland-soccer"
    result = bootstrap_consumer("scotland", tmp_path)

    assert result == target.resolve()
    assert 'get_league_config("scotland")' in (target / "config.py").read_text()
    workflow = (target / ".github" / "workflows" / "artifact-pipeline.yml").read_text()
    assert "league_key: scotland" in workflow
    assert "Scottish Premiership artifact pipeline" in workflow
    assert (target / "scripts" / "verify_consumer.py").is_file()
    assert (target / "precomputed" / ".gitkeep").is_file()
    assert not tuple(target.rglob("__pycache__"))


def test_bootstrap_refuses_to_overwrite_existing_path(tmp_path: Path):
    target = tmp_path / "netherlands-soccer"
    target.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        bootstrap_consumer("eredivisie", tmp_path)


def test_bootstrap_rejects_league_without_baseline_fixture_provider(tmp_path: Path):
    with pytest.raises(ValueError, match="not consumer-ready"):
        bootstrap_consumer("portugal", tmp_path)
