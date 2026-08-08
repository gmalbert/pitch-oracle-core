"""Shared Betting Oracle footer for all Streamlit league consumers."""

from __future__ import annotations

import streamlit as st


FOOTER_HTML = """
<div style="text-align:center; padding:20px 0; border-top:1px solid #e0e0e0; margin-top:40px;">
  <p style="margin:0 0 10px; font-size:14px; color:#666; font-family:sans-serif;">
    Powered by <a href="https://www.betting-oracle.com" target="_blank"
    style="color:#3b82f6; text-decoration:none; font-weight:bold;">Betting Oracle</a>
  </p>
  <p style="margin:0 0 15px; font-size:12px; color:#888; font-family:sans-serif;">
    Sports Prediction Analytics<br>
    All content is for informational purposes only and does not constitute betting advice. Wager responsibly.
  </p>
  <a href="https://www.betting-oracle.com" target="_blank">
    <img src="https://raw.githubusercontent.com/gmalbert/betting-oracle/main/data_files/logo.png"
    alt="Betting Oracle Logo" style="height:60px; width:auto; border:none;">
  </a>
</div>
"""


def render_footer() -> None:
    """Render the shared footer at the bottom of the current Streamlit page."""
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
