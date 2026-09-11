"""Tests for P1/P2 modules — markets, ratings, betting, walk-forward, etc."""

import numpy as np
import pandas as pd
import pytest


# ── P1.1: FootballProbabilityGrid wrapper ──────────────────────────────

class TestMarketGrid:
    def test_market_row_has_all_keys(self):
        from penaltyblog.models import DixonColesGoalModel
        from pitch_oracle_core.markets.grid import market_row
        gh = np.random.poisson(1.4, 60)
        ga = np.random.poisson(1.1, 60)
        th = np.random.choice(["A", "B", "C"], 60)
        ta = np.random.choice(["A", "B", "C"], 60)
        m = DixonColesGoalModel(gh, ga, th, ta)
        m.fit()
        grid = m.predict("A", "B", max_goals=10)
        row = market_row(grid, "test")
        assert "home_win" in row
        assert "btts_yes" in row
        assert "over_2_5" in row
        assert "ah_home_minus0_5_win" in row
        assert abs(row["home_win"] + row["draw"] + row["away_win"] - 1.0) < 1e-6

    def test_market_grid_from_lambdas(self):
        from pitch_oracle_core.markets.grid import market_grid_from_lambdas
        row = market_grid_from_lambdas(1.5, 1.2, rho=-0.05)
        assert 0 < row["home_win"] < 1
        assert abs(row["home_win"] + row["draw"] + row["away_win"] - 1.0) < 1e-6
        assert "score_grid_tail_mass" in row


# ── P1.4: Walk-forward evaluation ──────────────────────────────────────

class TestWalkForward:
    def test_walk_forward_returns_folds(self):
        from pitch_oracle_core.evaluation.walk_forward import walk_forward_evaluate
        from penaltyblog.models import PoissonGoalsModel
        np.random.seed(42)
        n = 300
        teams = ["A", "B", "C", "D", "E", "F"]
        df = pd.DataFrame({
            "team_home": np.random.choice(teams, n),
            "team_away": np.random.choice(teams, n),
            "goals_home": np.random.poisson(1.4, n),
            "goals_away": np.random.poisson(1.1, n),
            "datetime": pd.date_range("2020-01-01", periods=n, freq="5D"),
        })
        def factory(train):
            m = PoissonGoalsModel(
                train["goals_home"].to_numpy().copy(),
                train["goals_away"].to_numpy().copy(),
                train["team_home"].to_numpy().copy(),
                train["team_away"].to_numpy().copy(),
            )
            m.fit()
            return m
        result = walk_forward_evaluate(df, factory, window=120, horizon=30)
        assert result.n_folds > 0
        assert 0 < result.mean_brier < 1
        assert result.mean_rps > 0
        wf_df = result.to_dataframe()
        assert len(wf_df) == result.n_folds
        assert (wf_df["n_scored"] == wf_df["n_test"]).all()

    def test_walk_forward_accepts_date_column_and_includes_final_fold(self):
        from pitch_oracle_core.evaluation.walk_forward import walk_forward_evaluate
        from penaltyblog.models import PoissonGoalsModel
        n = 180
        teams = np.array(["A", "B", "C", "D"])
        df = pd.DataFrame({
            "team_home": np.resize(teams, n),
            "team_away": np.resize(teams[::-1], n),
            "goals_home": np.ones(n, dtype=int),
            "goals_away": np.zeros(n, dtype=int),
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        })

        def factory(train):
            model = PoissonGoalsModel(
                train.goals_home.to_numpy(), train.goals_away.to_numpy(),
                train.team_home.to_numpy(), train.team_away.to_numpy(),
            )
            model.fit()
            return model

        result = walk_forward_evaluate(df, factory, window=100, horizon=40)
        assert result.n_folds == 2


# ── P1.5/P1.6: Rating systems ─────────────────────────────────────────

class TestRatings:
    @pytest.fixture
    def sample_matches(self):
        np.random.seed(42)
        teams = ["Arsenal", "Chelsea", "Liverpool", "Man City"]
        n = 40
        return pd.DataFrame({
            "team_home": np.random.choice(teams, n),
            "team_away": np.random.choice(teams, n),
            "goals_home": np.random.poisson(1.4, n),
            "goals_away": np.random.poisson(1.1, n),
        })

    def test_elo_ratings(self, sample_matches):
        from pitch_oracle_core.ratings import compute_elo_ratings
        elo, history = compute_elo_ratings(sample_matches)
        assert len(history) == len(sample_matches)
        assert len(elo.ratings) > 0

    def test_massey_ratings(self, sample_matches):
        from pitch_oracle_core.ratings import compute_massey_ratings
        df = compute_massey_ratings(sample_matches)
        assert "team" in df.columns
        assert "rating" in df.columns

    def test_colley_ratings(self, sample_matches):
        from pitch_oracle_core.ratings import compute_colley_ratings
        df = compute_colley_ratings(sample_matches)
        assert "team" in df.columns

    def test_pi_ratings(self, sample_matches):
        from pitch_oracle_core.ratings import compute_pi_ratings
        pi, history = compute_pi_ratings(sample_matches)
        assert len(history) == len(sample_matches)

    def test_combined_rankings(self, sample_matches):
        from pitch_oracle_core.ratings import build_combined_rankings
        df = build_combined_rankings(sample_matches)
        assert "elo_rank" in df.columns
        assert "massey_rank" in df.columns
        assert "colley_rank" in df.columns
        assert "pi_rank" in df.columns

    def test_elo_form_delta(self, sample_matches):
        from pitch_oracle_core.ratings import compute_elo_ratings, elo_form_delta
        _, history = compute_elo_ratings(sample_matches)
        delta = elo_form_delta(history, "Arsenal", window_days=14)
        # May be None if not enough matches
        if delta is not None:
            assert isinstance(delta, float)

    def test_chronological_elo_uses_kickoff(self, sample_matches):
        from pitch_oracle_core.ratings import compute_elo_ratings

        ordered = sample_matches.copy()
        ordered["kickoff_utc"] = pd.date_range("2024-01-01", periods=len(ordered), tz="UTC")
        shuffled = ordered.sample(frac=1.0, random_state=11).reset_index(drop=True)
        _, history = compute_elo_ratings(shuffled)
        assert pd.Timestamp(history[0]["known_at"]) == pd.Timestamp("2024-01-01", tz="UTC")


# ── P2.5/P2.6/P2.7: Betting utilities ─────────────────────────────────

class TestBetting:
    def test_kelly_stake(self):
        from pitch_oracle_core.betting import compute_kelly_stake
        result = compute_kelly_stake(2.10, 0.55, fraction=0.5)
        assert "stake" in result
        assert "edge" in result

    def test_find_value_bets(self):
        from pitch_oracle_core.betting import find_value_bets
        bets = find_value_bets([0.55, 0.25, 0.20], [2.10, 3.40, 3.60])
        # Home has edge: 0.55 > implied ~0.46
        assert len(bets) > 0
        assert bets[0].outcome == "home"

    def test_find_arbitrage(self):
        from pitch_oracle_core.betting import find_arbitrage
        # No arb: normal odds
        arb = find_arbitrage([[2.10, 3.40, 3.60], [2.05, 3.50, 3.55]])
        assert isinstance(arb.has_arbitrage, bool)

    def test_compute_hedge(self):
        from pitch_oracle_core.betting import compute_hedge
        result = compute_hedge([100], [2.50], [1.80])
        assert "hedge_stakes" in result
        assert "guaranteed_profit" in result


# ── P2.8: Odds conversion ─────────────────────────────────────────────

class TestOddsConversion:
    def test_decimal_to_american(self):
        from pitch_oracle_core.odds_conversion import decimal_to_american
        assert decimal_to_american(2.50) == 150
        assert decimal_to_american(1.50) == -200

    def test_american_to_decimal(self):
        from pitch_oracle_core.odds_conversion import american_to_decimal
        assert abs(american_to_decimal(150) - 2.50) < 0.01
        assert abs(american_to_decimal(-200) - 1.50) < 0.01

    def test_fractional_to_decimal(self):
        from pitch_oracle_core.odds_conversion import fractional_to_decimal
        assert abs(fractional_to_decimal("5/2") - 3.50) < 0.01

    def test_detect_format(self):
        from pitch_oracle_core.odds_conversion import detect_odds_format
        assert detect_odds_format(2.50) == "decimal"
        assert detect_odds_format(150) == "american"
        assert detect_odds_format("5/2") == "fractional"


# ── P2.1: Model variants ──────────────────────────────────────────────

class TestModelVariants:
    @pytest.fixture
    def goals_frame(self):
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "team_home": np.random.choice(["A", "B", "C", "D"], n),
            "team_away": np.random.choice(["A", "B", "C", "D"], n),
            "goals_home": np.random.poisson(1.4, n),
            "goals_away": np.random.poisson(1.1, n),
            "date": pd.date_range("2020-01-01", periods=n, freq="5D"),
        })

    def test_fit_dixon_coles(self, goals_frame):
        from pitch_oracle_core.model_variants import fit_model_variant
        result = fit_model_variant(goals_frame, "dixon_coles")
        assert result.model_name == "dixon_coles"
        grid = result.model.predict("A", "B", max_goals=10)
        assert abs(grid.home_win + grid.draw + grid.away_win - 1.0) < 1e-6

    def test_fit_poisson(self, goals_frame):
        from pitch_oracle_core.model_variants import fit_model_variant
        result = fit_model_variant(goals_frame, "poisson")
        assert result.model_name == "poisson"

    def test_unknown_model_raises(self, goals_frame):
        from pitch_oracle_core.model_variants import fit_model_variant
        with pytest.raises(ValueError, match="Unknown model"):
            fit_model_variant(goals_frame, "nonexistent")


# ── P2.13: Drift monitor ──────────────────────────────────────────────

class TestDriftMonitor:
    def test_psi_stable(self):
        from pitch_oracle_core.evaluation.drift_monitor import population_stability_index
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0, 1, 1000)
        psi = population_stability_index(ref, cur)
        assert psi < 0.1  # same distribution → stable

    def test_psi_shifted(self):
        from pitch_oracle_core.evaluation.drift_monitor import population_stability_index
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(3, 1, 1000)  # shifted
        psi = population_stability_index(ref, cur)
        assert psi > 0.25  # significant shift

    def test_drift_severity(self):
        from pitch_oracle_core.evaluation.drift_monitor import drift_severity
        assert drift_severity(0.05) == "stable"
        assert drift_severity(0.15) == "watch"
        assert drift_severity(0.30) == "action_required"

    def test_monitor_feature_drift(self):
        from pitch_oracle_core.evaluation.drift_monitor import monitor_feature_drift
        ref = pd.DataFrame({"x": np.random.normal(0, 1, 200), "y": np.random.normal(0, 1, 200)})
        cur = pd.DataFrame({"x": np.random.normal(0, 1, 200), "y": np.random.normal(3, 1, 200)})
        findings = monitor_feature_drift(ref, cur, features=["x", "y"])
        assert len(findings) == 2
        y_finding = [f for f in findings if f.feature == "y"][0]
        assert y_finding.severity == "action_required"


# ── P2.14: Cohort slices ──────────────────────────────────────────────

class TestCohortSlices:
    def test_assign_cohorts(self):
        from pitch_oracle_core.evaluation.cohort_slices import assign_cohorts
        df = pd.DataFrame({
            "rest_days": [2, 4, 6, 8, 15],
            "is_promoted": [True, False, False, True, False],
            "is_derby": [False, True, False, False, True],
            "matchday": [5, 15, 30, 35, 40],
        })
        result = assign_cohorts(df)
        assert "cohort_rest" in result.columns
        assert "cohort_promoted" in result.columns
        assert "cohort_derby" in result.columns
        assert "cohort_phase" in result.columns


# ── P2.4: Margin audit ────────────────────────────────────────────────

class TestMarginAudit:
    def test_compute_bookmaker_margins(self):
        from pitch_oracle_core.margin_audit import compute_bookmaker_margins
        odds = pd.DataFrame({
            "bookmaker": ["Bet365", "Pinnacle"],
            "home": [2.10, 2.15],
            "draw": [3.40, 3.45],
            "away": [3.60, 3.55],
        })
        margins = compute_bookmaker_margins(odds)
        assert len(margins) == 2
        assert all(margins["margin"] > 0)


# ── P1.10: xT ─────────────────────────────────────────────────────────

class TestXT:
    def test_pretrained_xt_value(self):
        from pitch_oracle_core.xt import pretrained_xt_value
        val = pretrained_xt_value(85, 50)
        assert val > 0
        # Near penalty spot should have high xT
        val_near = pretrained_xt_value(95, 50)
        assert val_near >= val


# ── P2.16: Custom theme ───────────────────────────────────────────────

class TestPitchTheme:
    def test_theme_exists(self):
        from pitch_oracle_core.pitch_theme import PITCH_ORACLE_THEME, PITCH_ORACLE_LIGHT
        assert PITCH_ORACLE_THEME.name == "minimal"
        assert PITCH_ORACLE_LIGHT.name == "light"


# ── P2.15: xT comparison ──────────────────────────────────────────────

class TestXTComparison:
    def test_compare_rankings(self):
        from pitch_oracle_core.xt_comparison import compare_xt_rankings
        custom = pd.DataFrame({"player": ["A", "B", "C", "D", "E"], "xt_added_per90": [0.5, 0.4, 0.3, 0.2, 0.1]})
        pretrained = pd.DataFrame({"player": ["A", "B", "C", "D", "E"], "xt_added_per90": [0.48, 0.42, 0.28, 0.22, 0.09]})
        result = compare_xt_rankings(custom, pretrained)
        assert result["spearman_rho"] > 0.9  # highly correlated
        assert result["n_players"] == 5
