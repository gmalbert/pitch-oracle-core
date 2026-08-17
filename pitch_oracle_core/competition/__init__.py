"""Rule-aware standings, simulations, and season-stakes services."""

from .standings import calculate_table
from .phases import bracket_paths, generate_pool_fixtures, transition_phase

__all__ = [
    "bracket_paths", "calculate_table", "generate_pool_fixtures", "transition_phase",
]
