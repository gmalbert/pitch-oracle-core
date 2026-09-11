import numpy as np
import pandas as pd


def test_posterior_probability_draws_are_finite_and_bounded():
    from pitch_oracle_core.bayesian import posterior_probability_draws

    class FakeBayesianModel:
        n_teams = 2
        team_to_idx = {"A": 0, "B": 1}
        trace = np.array([
            [0.10, -0.10, -0.05, 0.05, 0.15, -0.04],
            [0.12, -0.12, -0.05, 0.05, 0.15, -0.04],
            [0.08, -0.08, -0.05, 0.05, 0.15, -0.04],
        ])

    draws = posterior_probability_draws(
        FakeBayesianModel(), "A", "B", max_goals=10, n_samples=10
    )
    assert draws.shape == (3, 3)
    assert np.isfinite(draws).all()
    assert (draws >= 0).all() and (draws <= 1).all()
    assert np.allclose(draws.sum(axis=1), 1.0, atol=1e-8)


def test_elo_asof_features_exclude_future_results():
    from pitch_oracle_core.ratings import compute_elo_ratings, elo_asof_features

    matches = pd.DataFrame({
        "team_home": ["A", "B"],
        "team_away": ["B", "A"],
        "goals_home": [3, 0],
        "goals_away": [0, 2],
        "date": pd.to_datetime(["2024-01-01", "2024-01-10"], utc=True),
    })
    _, history = compute_elo_ratings(matches)
    fixtures = pd.DataFrame({
        "team_home": ["A", "A"],
        "team_away": ["B", "B"],
        "issue_time": pd.to_datetime(["2024-01-01", "2024-01-20"], utc=True),
    })
    result = elo_asof_features(history, fixtures)
    assert pd.isna(result.loc[0, "elo_home_pre"])
    assert result.loc[1, "elo_home_pre"] != 1500.0


def test_native_dixon_coles_respects_neutral_venue():
    from pitch_oracle_core.models.dixon_coles import DixonColesModel

    frame = pd.DataFrame({
        "kickoff_utc": pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC"),
        "home_team_id": ["A", "B"] * 4,
        "away_team_id": ["B", "A"] * 4,
        "home_goals": [2, 0, 2, 1, 3, 0, 2, 1],
        "away_goals": [0, 1, 1, 0, 0, 2, 1, 0],
    })
    model = DixonColesModel().fit(frame, as_of="2024-02-01T00:00:00Z")
    home, away = model.expected_goals("A", "B")
    neutral_home, neutral_away = model.expected_goals("A", "B", neutral_venue=True)
    assert home > neutral_home
    assert away == neutral_away
