import numpy as np
import pandas as pd

from pitch_oracle_core.models import PenaltyBlogDixonColes
from pitch_oracle_core.models.protocol import FixtureFeatures


def _matches(n=48):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "team_home": rng.choice(["A", "B", "C", "D"], n),
        "team_away": rng.choice(["A", "B", "C", "D"], n),
        "goals_home": rng.poisson(1.4, n),
        "goals_away": rng.poisson(1.1, n),
        "date": pd.date_range("2024-01-01", periods=n, freq="7D"),
    })


def test_adapter_preserves_explicit_tail_mass():
    model = PenaltyBlogDixonColes(max_goals=8).fit(
        _matches(), cutoff_utc=pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime()
    )
    fixture = FixtureFeatures(
        fixture_id="f-1",
        kickoff_utc=pd.Timestamp("2025-01-08", tz="UTC").to_pydatetime(),
        home_team_id="A",
        away_team_id="B",
        values={},
    )
    grid = model.predict_grid(fixture)
    assert grid.mass.shape[0] == grid.mass.shape[1]
    assert grid.mass.shape[0] > 1
    assert np.isclose(grid.mass.sum() + grid.tail_mass, 1.0)
    assert grid.tail_mass >= 0


def test_adapter_rejects_missing_training_columns():
    try:
        PenaltyBlogDixonColes().fit(
            pd.DataFrame({"team_home": ["A"]}),
            cutoff_utc=pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime(),
        )
    except ValueError as exc:
        assert "Missing columns" in str(exc)
    else:
        raise AssertionError("missing columns should fail closed")
