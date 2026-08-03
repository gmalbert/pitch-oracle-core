from pathlib import Path

from streamlit.testing.v1 import AppTest

from pitch_oracle_core.cache import CacheRequirement, write_cache_manifest
from pitch_oracle_core.config import LeagueConfig, ThemeConfig
from pitch_oracle_core.theme import LAUNCH_THEMES

THEME_FIELDS = {"primary", "primary_dark", "sidebar", "page", "border", "muted"}


def test_palette_registry_covers_every_field():
    assert LAUNCH_THEMES
    for name, palette in LAUNCH_THEMES.items():
        assert set(palette) == THEME_FIELDS, f"{name} is missing fields"
        for value in palette.values():
            assert value.startswith("#") and len(value) == 7, f"{name} {value} is not a hex color"


def _css(app) -> str:
    return "\n".join(markdown.value for markdown in app.markdown if "<style>" in markdown.value)


def test_dark_palette_switches_text_vars_to_light(tmp_path, monkeypatch):
    artifact = tmp_path / "data" / "ready.txt"
    artifact.parent.mkdir()
    artifact.write_text("ready", encoding="utf-8")
    write_cache_manifest(
        tmp_path,
        requirements=(CacheRequirement("ready", "data/ready.txt"),),
        league="test",
    )
    entrypoint = _write_entrypoint(tmp_path)
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(entrypoint, default_timeout=30).run()
    assert not app.exception

    # Find a Nighttime palette and switch to it.
    dark_name = next(name for name in LAUNCH_THEMES if name.startswith("🌙"))
    app.sidebar.selectbox[0].set_value(dark_name).run()
    assert not app.exception

    css = _css(app)
    # Dark palettes must render light body text and dark cards.
    assert "--pitch-text: #e8ecf2" in css
    assert "--pitch-card: #14181f" in css
    assert "--pitch-header-bg: #1a2029" in css
    assert '[data-testid="stHeader"]' in css
    assert 'background: var(--pitch-page) !important' in css
    # The palette's own colors still apply.
    assert f"--pitch-primary: {LAUNCH_THEMES[dark_name]['primary']}" in css
    assert f"--pitch-page: {LAUNCH_THEMES[dark_name]['page']}" in css


def test_light_palette_keeps_dark_text_vars(tmp_path, monkeypatch):
    artifact = tmp_path / "data" / "ready.txt"
    artifact.parent.mkdir()
    artifact.write_text("ready", encoding="utf-8")
    write_cache_manifest(
        tmp_path,
        requirements=(CacheRequirement("ready", "data/ready.txt"),),
        league="test",
    )
    entrypoint = _write_entrypoint(tmp_path)
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(entrypoint, default_timeout=30).run()
    assert not app.exception

    light_name = next(name for name in LAUNCH_THEMES if name.startswith("☀️"))
    app.sidebar.selectbox[0].set_value(light_name).run()
    assert not app.exception

    css = _css(app)
    assert "--pitch-text: #31333f" in css
    assert "--pitch-card: #f8fafc" in css
    assert "--pitch-header-bg: #eef3f8" in css
    assert '[data-testid="stHeader"]' in css
    assert 'background-color: var(--pitch-page) !important' in css


def _write_entrypoint(tmp_path: Path) -> Path:
    entrypoint = tmp_path / "predictions.py"
    entrypoint.write_text(
        "from pitch_oracle_core.config import LeagueConfig, ThemeConfig\n"
        "from pitch_oracle_core.theme import LAUNCH_THEMES\n"
        "from pitch_oracle_core.app_factory import run\n"
        "run(LeagueConfig(\n"
        "    key='test',\n"
        "    display_name='Test League',\n"
        "    football_data_div='T0',\n"
        "    espn_slug='test.1',\n"
        "    clubelo_code='TEST_1',\n"
        "    team_count=18,\n"
        "    season_months=(8, 5),\n"
        "    theme=ThemeConfig(launch_theme_choices=tuple(LAUNCH_THEMES)),\n"
        "), root=r'{root}')\n".format(root=tmp_path),
        encoding="utf-8",
    )
    return entrypoint


def test_theme_chooser_applies_selected_palette(tmp_path, monkeypatch):
    artifact = tmp_path / "data" / "ready.txt"
    artifact.parent.mkdir()
    artifact.write_text("ready", encoding="utf-8")
    write_cache_manifest(
        tmp_path,
        requirements=(CacheRequirement("ready", "data/ready.txt"),),
        league="test",
    )
    entrypoint = _write_entrypoint(tmp_path)
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(entrypoint, default_timeout=30).run()

    assert not app.exception
    # The chooser is rendered in the sidebar.
    labels = [select.label for select in app.sidebar.selectbox]
    assert any("theme" in label.lower() for label in labels)
    # The default palette (first choice) is applied on first load.
    first = app.sidebar.selectbox[0]
    assert first.value == list(LAUNCH_THEMES)[0]
    combined_css = "\n".join(markdown.value for markdown in app.markdown if "<style>" in markdown.value)
    assert f"--pitch-primary: {LAUNCH_THEMES[list(LAUNCH_THEMES)[0]]['primary']}" in combined_css

    # Pick a different theme and re-run: the palette must change.
    selected = list(LAUNCH_THEMES)[1]
    first.set_value(selected).run()
    assert not app.exception
    combined_css = "\n".join(markdown.value for markdown in app.markdown if "<style>" in markdown.value)
    assert f"--pitch-primary: {LAUNCH_THEMES[selected]['primary']}" in combined_css
    assert f"--pitch-sidebar: {LAUNCH_THEMES[selected]['sidebar']}" in combined_css


def test_theme_chooser_absent_without_choices(tmp_path, monkeypatch):
    artifact = tmp_path / "data" / "ready.txt"
    artifact.parent.mkdir()
    artifact.write_text("ready", encoding="utf-8")
    write_cache_manifest(
        tmp_path,
        requirements=(CacheRequirement("ready", "data/ready.txt"),),
        league="test",
    )
    entrypoint = tmp_path / "predictions.py"
    entrypoint.write_text(
        "from pitch_oracle_core.config import LeagueConfig\n"
        "from pitch_oracle_core.app_factory import run\n"
        "run(LeagueConfig(\n"
        "    key='test',\n"
        "    display_name='Test League',\n"
        "    football_data_div='T0',\n"
        "    espn_slug='test.1',\n"
        "    clubelo_code='TEST_1',\n"
        "    team_count=18,\n"
        "    season_months=(8, 5),\n"
        "), root=r'{root}')\n".format(root=tmp_path),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(entrypoint, default_timeout=30).run()

    assert not app.exception
    assert not app.sidebar.selectbox
