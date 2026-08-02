"""Public Streamlit application factory for thin league repositories."""

from pathlib import Path

from .config import LeagueConfig
from .cache import validate_cache
from .runtime import Runtime


def run(config: LeagueConfig, root: str = ".") -> None:
    runtime = Runtime.for_league(config, root).apply()
    import os
    import streamlit as st

    os.environ["PITCH_ORACLE_DISPLAY_NAME"] = config.display_name
    manifest_path = Path(root) / "precomputed" / "cache_manifest.json"
    if not manifest_path.is_file():
        st.set_page_config(
            page_title=f"Pitch Oracle — {config.display_name} setup",
            page_icon="⚽",
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
    validate_cache(root, expected_league=config.key)

    # st.navigation requires one, and only one, page-config call in the entrypoint.
    st.set_page_config(
        page_title=f"Pitch Oracle — {config.display_name}",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    from .navigation import build_navigation
    from .theme import apply_theme

    apply_theme(config)
    navigation = build_navigation(config)
    navigation.run()

