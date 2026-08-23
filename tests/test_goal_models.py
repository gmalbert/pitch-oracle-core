"""Tests for pitch_oracle_core.goal_models — Dixon-Coles training with time-decay."""

import json
import numpy as np
import pandas as pd
import pytest

from pitch_oracle_core.goal_models import (
    DEFAULT_XI,
    fit_dixon_coles,
    goals_frame_from_historical,
    load_goal_model,
    save_goal_model,
    time_decay_weights,
    training_set_hash,
)


@pytest.fixture
def synthetic_goals():
    np.random.seed(42)
    teams = [
        "Arsenal", "Chelsea", "Liverpool", "Man City", "Tottenham",
        "Man United", "Newcastle", "Brighton", "West Ham", "Aston Villa",
    ]
    n = 200
    return pd.DataFrame({
        "team_home": np.random.choice(teams, n),
        "team_away": np.random.choice(teams, n),
        "goals_home": np.random.poisson(1.4, n),
        "goals_away": np.random.poisson(1.1, n),
        "date": pd.date_range("2020-01-01", periods=n, freq="5D"),
    })


class TestTimeDecayWeights:
    def test_weights_are_between_zero_and_one(self, synthetic_goals):
        w = time_decay_weights(synthetic_goals["date"])
        assert w.shape == (len(synthetic_goals),)
        assert (w > 0).all() and (w <= 1).all()

    def test_more_recent_matches_have_higher_weights(self, synthetic_goals):
        w = time_decay_weights(synthetic_goals["date"])
        assert w[-1] >= w[0]

    def test_custom_xi_changes_decay_rate(self, synthetic_goals):
        w_fast = time_decay_weights(synthetic_goals["date"], xi=0.01)
        w_slow = time_decay_weights(synthetic_goals["date"], xi=0.0001)
        assert w_fast[0] < w_slow[0]


class TestTrainingSetHash:
    def test_deterministic(self, synthetic_goals):
        cols = ["goals_home", "goals_away", "team_home", "team_away"]
        assert training_set_hash(synthetic_goals[cols]) == training_set_hash(synthetic_goals[cols])

    def test_changes_on_data_change(self, synthetic_goals):
        cols = ["goals_home", "goals_away", "team_home", "team_away"]
        modified = synthetic_goals.copy()
        modified.loc[0, "goals_home"] = 99
        assert training_set_hash(synthetic_goals[cols]) != training_set_hash(modified[cols])


class TestFitDixonColes:
    def test_model_has_rho_parameter(self, synthetic_goals):
        model = fit_dixon_coles(synthetic_goals)
        params = model.get_params()
        assert "rho" in params

    def test_predict_grid_sums_to_one(self, synthetic_goals):
        model = fit_dixon_coles(synthetic_goals)
        grid = model.predict("Arsenal", "Chelsea", max_goals=10)
        total = grid.home_win + grid.draw + grid.away_win
        assert abs(total - 1.0) < 1e-6

    def test_predict_many_returns_grids(self, synthetic_goals):
        model = fit_dixon_coles(synthetic_goals)
        home_teams = np.array(["Arsenal", "Chelsea"])
        away_teams = np.array(["Chelsea", "Arsenal"])
        grids = model.predict_many(home_teams, away_teams, max_goals=10)
        assert len(grids) == 2
        for g in grids:
            assert abs(g.home_win + g.draw + g.away_win - 1.0) < 1e-6


class TestSaveLoadGoalModel:
    def test_round_trip(self, synthetic_goals, tmp_path):
        model = fit_dixon_coles(synthetic_goals)
        result = save_goal_model(model, synthetic_goals, tmp_path)
        assert result.model_path.exists()
        assert result.metadata_path.exists()
        assert result.n_fixtures == len(synthetic_goals)
        assert result.training_set_hash

        loaded = load_goal_model(result.model_path)
        grid_orig = model.predict("Arsenal", "Chelsea", max_goals=10)
        grid_loaded = loaded.predict("Arsenal", "Chelsea", max_goals=10)
        assert abs(grid_orig.home_win - grid_loaded.home_win) < 1e-10

    def test_metadata_json_is_valid(self, synthetic_goals, tmp_path):
        model = fit_dixon_coles(synthetic_goals)
        result = save_goal_model(model, synthetic_goals, tmp_path)
        meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        assert meta["model_type"] == "DixonColesGoalModel"
        assert meta["xi"] == DEFAULT_XI
        assert meta["n_fixtures"] == len(synthetic_goals)
        assert "rho" in meta["params"]


class TestGoalsFrameFromHistorical:
    def test_renames_and_types(self):
        df = pd.DataFrame({
            "HomeTeam": ["A", "B"], "AwayTeam": ["B", "A"],
            "FTHG": [2, 1], "FTAG": [0, 3], "Date": ["2024-01-01", "2024-01-08"],
        })
        frame = goals_frame_from_historical(df)
        assert list(frame.columns) == ["team_home", "team_away", "goals_home", "goals_away", "date"]
        assert frame["goals_home"].dtype == int

    def test_missing_column_raises(self):
        df = pd.DataFrame({"HomeTeam": ["A"], "AwayTeam": ["B"]})
        with pytest.raises(ValueError, match="misses"):
            goals_frame_from_historical(df)
