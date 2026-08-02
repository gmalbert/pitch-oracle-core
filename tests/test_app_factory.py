from pathlib import Path

from streamlit.testing.v1 import AppTest

from pitch_oracle_core.cache import CacheRequirement, write_cache_manifest


def test_non_epl_app_starts_without_legacy_epl_content(tmp_path, monkeypatch):
    artifact = tmp_path / "data" / "ready.txt"
    artifact.parent.mkdir()
    artifact.write_text("ready", encoding="utf-8")
    write_cache_manifest(
        tmp_path,
        requirements=(CacheRequirement("ready", "data/ready.txt"),),
        league="eredivisie",
    )
    entrypoint = tmp_path / "predictions.py"
    entrypoint.write_text(
        "from pitch_oracle_core import get_league_config, run_app\n"
        f"run_app(get_league_config('eredivisie'), root={str(tmp_path)!r})\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(entrypoint, default_timeout=30).run()

    assert not app.exception
    assert [title.value for title in app.title] == ["Eredivisie - Overview"]
    assert all("Premier League" not in element.value for element in app.markdown)


def test_missing_cache_shows_setup_screen_instead_of_crashing(tmp_path, monkeypatch):
    entrypoint = tmp_path / "predictions.py"
    entrypoint.write_text(
        "from pitch_oracle_core import get_league_config, run_app\n"
        f"run_app(get_league_config('eredivisie'), root={str(tmp_path)!r})\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(entrypoint, default_timeout=30).run()

    assert not app.exception
    assert [title.value for title in app.title] == ["Eredivisie setup required"]
    assert "Prediction artifacts have not been generated yet." in [warning.value for warning in app.warning]
