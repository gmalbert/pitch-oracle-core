"""Streamlit entrypoint for the league consumer."""

from config import LEAGUE_CONFIG
from pitch_oracle_core import run_app


run_app(LEAGUE_CONFIG)
