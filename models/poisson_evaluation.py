"""Shared evaluation utilities for the Poisson goals predictor."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, brier_score_loss

from .poisson_predictor import PoissonPredictor


def walk_forward_expectations(
    df: pd.DataFrame,
    prior_rate: float = 1.4,
    *,
    league_prior_matches: float = 20.0,
    team_prior_matches: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, float]]]:
    """Generate point-in-time Poisson forecasts with stable empirical priors."""
    required = {
        'HomeTeam', 'AwayTeam', 'FullTimeHomeGoals', 'FullTimeAwayGoals',
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    working = df.copy()
    if 'MatchDate' in working.columns:
        working['_evaluation_date'] = pd.to_datetime(working['MatchDate'], errors='coerce')
        working = working.sort_values('_evaluation_date', kind='stable')

    home_for: dict[str, list[float]] = {}
    home_against: dict[str, list[float]] = {}
    away_for: dict[str, list[float]] = {}
    away_against: dict[str, list[float]] = {}
    total_goals = 0.0
    matches_seen = 0
    predictor = PoissonPredictor()
    home_expected, away_expected, outcome_probabilities = [], [], []

    if prior_rate <= 0 or league_prior_matches <= 0 or team_prior_matches <= 0:
        raise ValueError("Poisson prior rates and weights must be positive")

    def average(store: dict[str, list[float]], team: str, fallback: float) -> float:
        total, count = store.get(team, [0.0, 0.0])
        return (total + fallback * team_prior_matches) / (count + team_prior_matches)

    def update(store: dict[str, list[float]], team: str, value: float) -> None:
        totals = store.setdefault(team, [0.0, 0.0])
        totals[0] += value
        totals[1] += 1

    for _, row in working.iterrows():
        home, away = str(row['HomeTeam']), str(row['AwayTeam'])
        league_rate = (
            total_goals + 2 * prior_rate * league_prior_matches
        ) / (2 * (matches_seen + league_prior_matches))
        predictor.league_avg_goals = league_rate
        home_rate, away_rate = predictor.estimate_goals(
            average(home_for, home, league_rate),
            average(home_against, home, league_rate),
            average(away_for, away, league_rate),
            average(away_against, away, league_rate),
        )
        scorelines = predictor.poisson_scoreline_probabilities(home_rate, away_rate, max_goals=10)
        home_expected.append(home_rate)
        away_expected.append(away_rate)
        outcome_probabilities.append(predictor.predict_match_outcome(scorelines))

        home_goals = float(row['FullTimeHomeGoals'])
        away_goals = float(row['FullTimeAwayGoals'])
        update(home_for, home, home_goals)
        update(home_against, home, away_goals)
        update(away_for, away, away_goals)
        update(away_against, away, home_goals)
        total_goals += home_goals + away_goals
        matches_seen += 1

    return np.asarray(home_expected), np.asarray(away_expected), outcome_probabilities


def predict_upcoming_outcomes(
    historical: pd.DataFrame,
    upcoming: pd.DataFrame,
    prior_rate: float = 1.4,
    *,
    league_prior_matches: float = 20.0,
    team_prior_matches: float = 5.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float]]]:
    """Return walk-forward Poisson 1X2 probabilities for upcoming fixtures.

    Team attack and defense rates are accumulated from completed history only,
    matching the discipline of ``walk_forward_expectations``: an upcoming
    fixture's forecast never sees a completed match dated after it.
    """
    required = {
        'HomeTeam', 'AwayTeam', 'FullTimeHomeGoals', 'FullTimeAwayGoals',
    }
    missing = required.difference(historical.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if 'MatchDate' in historical.columns:
        historical = historical.sort_values('MatchDate', kind='stable')

    home_for: dict[str, list[float]] = {}
    home_against: dict[str, list[float]] = {}
    away_for: dict[str, list[float]] = {}
    away_against: dict[str, list[float]] = {}
    total_goals = 0.0
    matches_seen = 0
    predictor = PoissonPredictor()

    if prior_rate <= 0 or league_prior_matches <= 0 or team_prior_matches <= 0:
        raise ValueError("Poisson prior rates and weights must be positive")

    def average(store: dict[str, list[float]], team: str, fallback: float) -> float:
        totals = store.get(team, [0.0, 0.0])
        return (totals[0] + fallback * team_prior_matches) / (totals[1] + team_prior_matches)

    def update(store: dict[str, list[float]], team: str, value: float) -> None:
        totals = store.setdefault(team, [0.0, 0.0])
        totals[0] += value
        totals[1] += 1

    for _, row in historical.iterrows():
        home, away = str(row['HomeTeam']), str(row['AwayTeam'])
        league_rate = (
            total_goals + 2 * prior_rate * league_prior_matches
        ) / (2 * (matches_seen + league_prior_matches))
        home_goals = float(row['FullTimeHomeGoals'])
        away_goals = float(row['FullTimeAwayGoals'])
        update(home_for, home, home_goals)
        update(home_against, home, away_goals)
        update(away_for, away, away_goals)
        update(away_against, away, home_goals)
        total_goals += home_goals + away_goals
        matches_seen += 1

    league_rate = (
        total_goals + 2 * prior_rate * league_prior_matches
    ) / (2 * (matches_seen + league_prior_matches))
    predictor.league_avg_goals = league_rate

    outcome_probabilities: list[tuple[float, float, float]] = []
    expected_goals: list[tuple[float, float]] = []
    for _, row in upcoming.iterrows():
        home, away = str(row['HomeTeam']), str(row['AwayTeam'])
        home_rate, away_rate = predictor.estimate_goals(
            average(home_for, home, league_rate),
            average(home_against, home, league_rate),
            average(away_for, away, league_rate),
            average(away_against, away, league_rate),
        )
        scorelines = predictor.poisson_scoreline_probabilities(home_rate, away_rate, max_goals=10)
        outcome_probabilities.append(predictor.predict_match_outcome(scorelines))
        expected_goals.append((float(home_rate), float(away_rate)))

    return outcome_probabilities, expected_goals


def evaluate_poisson_dataframe(df: pd.DataFrame) -> dict:
    """Evaluate a PoissonPredictor on a historical dataframe.

    Args:
        df: Historical matches containing teams, final goals, and result.

    Returns:
        dict with keys:
            league_avg, home_mae, away_mae, home_rmse, away_rmse,
            outcome_acc, brier_home, brier_draw, brier_away
    """
    required = ['HomeTeam', 'AwayTeam', 'FullTimeHomeGoals', 'FullTimeAwayGoals', 'FullTimeResult']
    df = df.dropna(subset=required).copy()
    df = df[df['FullTimeResult'].isin({'H', 'D', 'A'})]
    if 'MatchDate' in df.columns:
        df['_evaluation_date'] = pd.to_datetime(df['MatchDate'], errors='coerce')
        df = df.sort_values('_evaluation_date', kind='stable')
    league_avg = (df['FullTimeHomeGoals'] + df['FullTimeAwayGoals']).mean() / 2
    home_exp, away_exp, outcome_probs = walk_forward_expectations(df)
    pred_outcome = np.argmax(np.vstack(outcome_probs), axis=1)

    y_home = df['FullTimeHomeGoals'].values
    y_away = df['FullTimeAwayGoals'].values
    y_outcome = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values

    y_home_win = (y_outcome == 0).astype(int)
    y_draw = (y_outcome == 1).astype(int)
    y_away_win = (y_outcome == 2).astype(int)
    prob_array = np.vstack(outcome_probs)

    metrics = {
        'league_avg': league_avg,
        'home_mae': mean_absolute_error(y_home, home_exp),
        'away_mae': mean_absolute_error(y_away, away_exp),
        'home_rmse': np.sqrt(mean_squared_error(y_home, home_exp)),
        'away_rmse': np.sqrt(mean_squared_error(y_away, away_exp)),
        'outcome_acc': accuracy_score(y_outcome, pred_outcome),
        'brier_home': brier_score_loss(y_home_win, prob_array[:, 0]),
        'brier_draw': brier_score_loss(y_draw, prob_array[:, 1]),
        'brier_away': brier_score_loss(y_away_win, prob_array[:, 2]),
    }
    from pitch_oracle_core.model_audit import probability_metrics
    metrics.update({
        f"outcome_{name}": value
        for name, value in probability_metrics(y_outcome, prob_array).items()
    })
    return metrics


def evaluate_poisson_file(path: str) -> dict:
    """Convenience wrapper to load a csv and evaluate it."""
    df = pd.read_csv(path, sep='\t')
    return evaluate_poisson_dataframe(df)
