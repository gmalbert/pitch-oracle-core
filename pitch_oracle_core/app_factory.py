"""Public Streamlit application factory for thin league repositories."""

from pathlib import Path
import json

from .config import LeagueConfig
from .cache import validate_cache
from .runtime import Runtime


def _browser_title(config: LeagueConfig) -> str:
    """Return the consumer-facing browser title for a league app."""
    country = config.country_name or config.key.title()
    return f"{config.display_name} ({country} soccer)"


def run(config: LeagueConfig, root: str = ".", *, scenario_adapter=None) -> None:
    runtime = Runtime.for_league(config, root).apply()
    import os
    import streamlit as st

    os.environ["PITCH_ORACLE_DISPLAY_NAME"] = config.display_name
    manifest_path = Path(root) / "precomputed" / "cache_manifest.json"
    if not manifest_path.is_file():
        st.set_page_config(
            page_title=f"{_browser_title(config)} setup",
            page_icon=config.country_flag,
            layout="wide",
        )
        st.title(f"{config.display_name} setup required")
        st.warning("Prediction artifacts have not been generated yet.")
        st.markdown(
            "Run the league's **artifact pipeline** from GitHub Actions, or follow "
            "the First run steps in this repository's README. Once the pipeline "
            "commits `precomputed/cache_manifest.json`, reload this page."
        )
        return
    from .artifacts.repository import ArtifactRepository

    repository = ArtifactRepository.from_manifest(root, expected_league=config.key)
    manifest = repository.manifest
    if manifest.get("schema_version") != 3:
        validate_cache(root, expected_league=config.key)

    # st.navigation requires one, and only one, page-config call in the entrypoint.
    st.set_page_config(
        page_title=_browser_title(config),
        page_icon=config.country_flag,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    from .theme import apply_theme

    apply_theme(config)
    if manifest.get("schema_version") == 3:
        from .ui.app import run_navigation

        run_navigation(config, root, scenario_adapter=scenario_adapter)
    else:
        from .navigation import build_navigation

        navigation = build_navigation(config)
        navigation.run()

