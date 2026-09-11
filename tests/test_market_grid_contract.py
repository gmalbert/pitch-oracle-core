import numpy as np

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid
from pitch_oracle_core.markets.grid import market_row_from_domain


def test_domain_market_surface_has_consistent_outcomes():
    mass = np.zeros((4, 4), dtype=float)
    mass[1, 0] = 0.25
    mass[1, 1] = 0.25
    mass[0, 1] = 0.25
    mass[2, 2] = 0.25
    row = market_row_from_domain(ProbabilityGrid(mass, 0.0, 3, 3))
    assert np.isclose(row["home_win"] + row["draw"] + row["away_win"], 1.0)
    assert np.isclose(row["btts_yes"] + row["btts_no"], 1.0)
    assert np.isclose(row["ah_home_minus0_5_win"], row["home_win"])
    assert np.isclose(row["score_grid_tail_mass"], 0.0)
