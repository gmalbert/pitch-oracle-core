"""Public Streamlit application factory for thin league repositories."""

from .config import LeagueConfig
from .runtime import Runtime


def run(config: LeagueConfig, root: str = ".") -> None:
    runtime = Runtime.for_league(config, root).apply()
    import os
    os.environ["PITCH_ORACLE_DISPLAY_NAME"] = config.display_name
    import app_shell  # noqa: F401

