from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pitch_oracle_core.cache import CacheRequirement, write_cache_manifest
from pitch_oracle_core.config import ThemeConfig
from pitch_oracle_core.theme import (
    DAY_THEME_NAME,
    LAUNCH_THEMES,
    NIGHT_THEME_NAME,
    _browser_local_hour,
    _theme_name_for_hour,
)


THEME_FIELDS = {"primary", "primary_dark", "sidebar", "page", "border", "muted"}


def _css(app) -> str:
    return "\n".join(markdown.value for markdown in app.markdown if "<style>" in markdown.value)


def _write_entrypoint(tmp_path: Path, local_hour: int) -> Path:
    entrypoint = tmp_path / "predictions.py"
    entrypoint.write_text(
        "from pitch_oracle_core.config import LeagueConfig\n"
        "from pitch_oracle_core import theme\n"
        "from pitch_oracle_core.app_factory import run\n"
        f"theme._browser_local_hour = lambda: {local_hour}\n"
        "run(LeagueConfig(\n"
        "    key='test',\n"
        "    display_name='Test League',\n"
        "    football_data_div='T0',\n"
        "    espn_slug='test.1',\n"
        "    clubelo_code='TEST_1',\n"
        "    team_count=18,\n"
        "    season_months=(8, 5),\n"
        f"), root=r'{tmp_path}')\n",
        encoding="utf-8",
    )
    return entrypoint


def _run_app(tmp_path: Path, monkeypatch, local_hour: int):
    artifact = tmp_path / "data" / "ready.txt"
    artifact.parent.mkdir()
    artifact.write_text("ready", encoding="utf-8")
    write_cache_manifest(
        tmp_path,
        requirements=(CacheRequirement("ready", "data/ready.txt"),),
        league="test",
    )
    entrypoint = _write_entrypoint(tmp_path, local_hour)
    monkeypatch.chdir(tmp_path)
    return AppTest.from_file(entrypoint, default_timeout=30).run()


def test_production_palettes_cover_every_theme_field():
    for name in (DAY_THEME_NAME, NIGHT_THEME_NAME):
        palette = LAUNCH_THEMES[name]
        assert set(palette) == THEME_FIELDS
        for value in palette.values():
            assert value.startswith("#") and len(value) == 7


def test_legacy_theme_choices_remain_accepted_but_are_not_used():
    assert ThemeConfig(launch_theme_choices=("Old chooser value",)).launch_theme_choices


@pytest.mark.parametrize(
    ("hour", "expected"),
    ((0, NIGHT_THEME_NAME), (6, NIGHT_THEME_NAME), (7, DAY_THEME_NAME),
     (18, DAY_THEME_NAME), (19, NIGHT_THEME_NAME), (23, NIGHT_THEME_NAME)),
)
def test_theme_follows_browser_local_hour(hour, expected):
    assert _theme_name_for_hour(hour) == expected


def test_theme_rejects_invalid_hour():
    with pytest.raises(ValueError, match="between 0 and 23"):
        _theme_name_for_hour(24)


def test_browser_timezone_and_offset_are_supported():
    noon_utc = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)

    assert _browser_local_hour(now_utc=noon_utc, timezone_name="America/New_York") == 7
    assert _browser_local_hour(
        now_utc=noon_utc,
        timezone_name="Invalid/Timezone",
        timezone_offset_minutes=300,
    ) == 7


def test_daytime_app_uses_delft_blue_without_chooser(tmp_path, monkeypatch):
    app = _run_app(tmp_path, monkeypatch, local_hour=12)

    assert not app.exception
    assert not app.sidebar.selectbox
    css = _css(app)
    palette = LAUNCH_THEMES[DAY_THEME_NAME]
    assert f"--pitch-primary: {palette['primary']}" in css
    assert f"--pitch-page: {palette['page']}" in css
    assert "--pitch-text: #31333f" in css


def test_nighttime_app_uses_winter_night_without_chooser(tmp_path, monkeypatch):
    app = _run_app(tmp_path, monkeypatch, local_hour=22)

    assert not app.exception
    assert not app.sidebar.selectbox
    css = _css(app)
    palette = LAUNCH_THEMES[NIGHT_THEME_NAME]
    assert f"--pitch-primary: {palette['primary']}" in css
    assert f"--pitch-page: {palette['page']}" in css
    assert "--pitch-text: #e8ecf2" in css
    assert "--pitch-card: #14181f" in css
    assert '[data-testid="stDownloadButton"] > button' in css
    assert '[data-testid="stButtonGroup"] [data-variant="segmented_control"]' in css
    assert '[data-testid="stMain"] [data-testid="stImage"]' not in css
