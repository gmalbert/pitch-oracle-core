"""
Ensemble Model Predictor for Premier League Match Outcomes

Combines multiple machine learning models for more robust predictions.
"""

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import numpy as np


def create_ensemble_model():
    """Create ensemble of multiple classifiers"""

    # Individual models
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        random_state=42
    )

    lr = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    # Voting ensemble (soft voting for probabilities)
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf', rf),
            ('gb', gb),
            ('lr', lr)
        ],
        voting='soft',
        weights=[2, 1.5, 1, 0.5]  # Higher weight for XGB
    )

    return ensemble


def create_simple_ensemble():
    """Create a simpler ensemble for faster training/testing"""
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )

    # Simple ensemble with just XGBoost and Random Forest
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf', rf)
        ],
        voting='soft',
        weights=[2, 1]  # Higher weight for XGB
    )

    return ensemble