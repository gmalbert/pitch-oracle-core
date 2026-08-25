"""Custom theme for pb.viz.Pitch — one Theme instance reused by every page."""

from __future__ import annotations

from penaltyblog.viz import Theme

# The canonical Pitch Oracle theme — uses penaltyblog's built-in theme names.
# Available themes: "minimal", "dark", "light", "default"
PITCH_ORACLE_THEME = Theme(name="minimal")
PITCH_ORACLE_LIGHT = Theme(name="light")
