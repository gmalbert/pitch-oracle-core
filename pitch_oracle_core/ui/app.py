"""Manifest-v3 Streamlit application entrypoint."""

from pathlib import Path
from dataclasses import replace
import streamlit as st

from .context import AppContext
from .navigation import build_navigation
from .repository import ArtifactRepository
from .formatters import timezone_control


def build_context(config, root: str | Path, *, scenario_adapter=None) -> AppContext:
    repository = ArtifactRepository.from_manifest(root, expected_league=config.key)
    manifest = repository.manifest
    return AppContext(
        config=config,
        repository=repository,
        capabilities=repository.capabilities,
        edition_id=str(manifest.get("edition_id", "unknown edition")),
        scenario_adapter=scenario_adapter,
        display_timezone=config.sources.weather_timezone,
    )


def run_navigation(config, root: str | Path = ".", *, scenario_adapter=None) -> None:
    context = build_context(config, root, scenario_adapter=scenario_adapter)
    st.sidebar.markdown(f"## {config.country_flag} Pitch Oracle")
    st.sidebar.caption(f"{config.display_name} · {context.edition_id}")
    st.sidebar.caption(
        "Serving artifact "
        f"{context.repository.manifest.get('producer_version', 'unknown version')} · "
        f"generated {context.repository.manifest.get('generated_at', 'unknown time')}"
    )
    if context.repository.manifest.get("serving_fallback"):
        st.sidebar.warning("Serving the last valid same-edition artifact graph.")
    context = replace(
        context,
        display_timezone=timezone_control(config.sources.weather_timezone),
    )
    build_navigation(context).run()
