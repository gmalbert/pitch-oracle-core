"""Regularized, probability-first classifier for the production no-odds model."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def create_no_odds_classifier():
    """Return the low-variance model selected by rolling-origin validation."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.2,
            max_iter=2000,
            random_state=42,
        ),
    )
