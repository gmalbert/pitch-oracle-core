"""Thin Streamlit entrypoint helper for league repositories."""

from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.runtime import Runtime


def run(config: LeagueConfig, root: str = ".") -> None:
    runtime = Runtime.for_league(config, root).apply()
    import os
    os.environ["PITCH_ORACLE_DISPLAY_NAME"] = config.display_name
    # app_shell is imported only after the league runtime is installed because its
    # Streamlit/data paths are initialized at import time.
    import app_shell  # noqa: F401

