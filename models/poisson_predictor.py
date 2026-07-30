"""
Poisson Regression Model for Football Goal Prediction

This module implements Poisson distribution-based goal prediction for football matches.
It estimates expected goals using team attacking/defensive strengths and calculates
probabilities for different scorelines and match outcomes.
"""

from scipy.stats import poisson
import numpy as np
import pandas as pd
import pickle
import os
from os import path


class PoissonPredictor:
    """Poisson regression model for football goal prediction"""

    def __init__(self):
        self.league_avg_goals = 1.4  # Typical Premier League average
        self.is_trained = False

    def estimate_goals(self, home_attack, home_defense, away_attack, away_defense, league_avg=None):
        """
        Estimate expected goals using team strengths

        Args:
            home_attack: Home team attacking strength (goals per game)
            home_defense: Home team defensive strength (goals conceded per game)
            away_attack: Away team attacking strength (goals per game)
            away_defense: Away team defensive strength (goals conceded per game)
            league_avg: League average goals per match (optional)

        Returns:
            tuple: (home_expected_goals, away_expected_goals)
        """
        if league_avg is None:
            league_avg = self.league_avg_goals

        # Convert defense to strength (inverse of goals conceded)
        home_def_strength = 1 / (home_defense + 0.1)  # Add small constant to avoid division by zero
        away_def_strength = 1 / (away_defense + 0.1)

        home_expected = league_avg * home_attack * away_def_strength
        away_expected = league_avg * away_attack * home_def_strength

        return home_expected, away_expected

    def poisson_scoreline_probabilities(self, home_exp, away_exp, max_goals=5):
        """
        Calculate probability matrix for all possible scorelines

        Args:
            home_exp: Expected goals for home team
            away_exp: Expected goals for away team
            max_goals: Maximum number of goals to consider (default: 5)

        Returns:
            numpy.ndarray: Matrix of scoreline probabilities
        """
        scoreline_probs = np.zeros((max_goals + 1, max_goals + 1))

        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob_home = poisson.pmf(home_goals, home_exp)
                prob_away = poisson.pmf(away_goals, away_exp)
                scoreline_probs[home_goals, away_goals] = prob_home * prob_away

        return scoreline_probs

    def predict_match_outcome(self, scoreline_probs):
        """
        Convert scoreline probabilities to match outcome probabilities

        Args:
            scoreline_probs: Matrix of scoreline probabilities

        Returns:
            tuple: (home_win_prob, draw_prob, away_win_prob)
        """
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0

        rows, cols = scoreline_probs.shape

        for home_goals in range(rows):
            for away_goals in range(cols):
                prob = scoreline_probs[home_goals, away_goals]

                if home_goals > away_goals:
                    home_win_prob += prob
                elif home_goals == away_goals:
                    draw_prob += prob
                else:
                    away_win_prob += prob

        return home_win_prob, draw_prob, away_win_prob

    def predict_with_poisson(self, home_team, away_team, team_stats_df):
        """
        Predict match using Poisson regression

        Args:
            home_team: Home team name
            away_team: Away team name
            team_stats_df: DataFrame with team statistics

        Returns:
            dict: Prediction results including probabilities and expected goals
        """
        try:
            # Get team stats
            home_data = team_stats_df[team_stats_df['Team'] == home_team]
            away_data = team_stats_df[team_stats_df['Team'] == away_team]

            if len(home_data) == 0 or len(away_data) == 0:
                return {
                    'error': f'Team statistics not found for {home_team} or {away_team}',
                    'HomeWinProb': 0.33,
                    'DrawProb': 0.34,
                    'AwayWinProb': 0.33
                }

            home_data = home_data.iloc[0]
            away_data = away_data.iloc[0]

            # Extract relevant stats (adjust column names as needed)
            home_attack = home_data.get('HomeGoalsAve', home_data.get('AvgHomeGoals', 1.5))
            home_defense = home_data.get('AwayGoalsConcededAve', home_data.get('AvgAwayGoalsConceded', 1.2))
            away_attack = away_data.get('AwayGoalsAve', away_data.get('AvgAwayGoals', 1.3))
            away_defense = away_data.get('HomeGoalsConcededAve', away_data.get('AvgHomeGoalsConceded', 1.1))

            # Calculate expected goals
            home_exp, away_exp = self.estimate_goals(
                home_attack=home_attack,
                home_defense=home_defense,
                away_attack=away_attack,
                away_defense=away_defense
            )

            # Get scoreline probabilities
            scorelines = self.poisson_scoreline_probabilities(home_exp, away_exp)

            # Convert to match outcome
            home_win_prob, draw_prob, away_win_prob = self.predict_match_outcome(scorelines)

            # Find most likely scoreline
            most_likely_idx = np.unravel_index(scorelines.argmax(), scorelines.shape)
            most_likely_score = f"{most_likely_idx[0]}-{most_likely_idx[1]}"

            return {
                'HomeWinProb': home_win_prob,
                'DrawProb': draw_prob,
                'AwayWinProb': away_win_prob,
                'ExpectedHomeGoals': home_exp,
                'ExpectedAwayGoals': away_exp,
                'MostLikelyScore': most_likely_score,
                'ScorelineProbabilities': scorelines
            }

        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'HomeWinProb': 0.33,
                'DrawProb': 0.34,
                'AwayWinProb': 0.33
            }

    def save_model(self, filepath):
        """Save the Poisson predictor to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath):
        """Load a Poisson predictor from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def create_poisson_predictor():
    """Factory function to create a Poisson predictor"""
    return PoissonPredictor()


# Convenience functions for integration
def predict_match_poisson(home_team, away_team, team_stats_df):
    """
    Convenience function for single match prediction

    Args:
        home_team: Home team name
        away_team: Away team name
        team_stats_df: DataFrame with team statistics

    Returns:
        dict: Prediction results
    """
    predictor = PoissonPredictor()
    return predictor.predict_with_poisson(home_team, away_team, team_stats_df)


if __name__ == "__main__":
    # Example usage
    print("Poisson Predictor for Football Goals")
    print("=" * 40)

    # Create sample team stats
    sample_stats = pd.DataFrame({
        'Team': ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester City'],
        'HomeGoalsAve': [2.1, 1.8, 2.3, 2.5],
        'AwayGoalsAve': [1.7, 1.5, 2.0, 2.2],
        'HomeGoalsConcededAve': [0.9, 1.1, 0.8, 0.7],
        'AwayGoalsConcededAve': [1.2, 1.3, 1.0, 0.9]
    })

    predictor = PoissonPredictor()

    # Test prediction
    result = predictor.predict_with_poisson('Arsenal', 'Chelsea', sample_stats)

    if 'error' not in result:
        print(f"Arsenal vs Chelsea Prediction:")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".2f")
        print(".2f")
        print(f"Most Likely Score: {result['MostLikelyScore']}")
    else:
        print(f"Error: {result['error']}")