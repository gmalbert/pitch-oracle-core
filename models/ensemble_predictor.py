"""
Ensemble Model Predictor for Premier League Match Outcomes

Combines multiple machine learning models for more robust predictions.
"""

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def create_ensemble_model():
    """Create ensemble of multiple classifiers"""

    # Individual models
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=1,
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=1,
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=5,
        random_state=42,
    )

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, n_jobs=1),
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
        weights=[1, 1, 1, 1],
        n_jobs=1,
    )

    return ensemble


def create_simple_ensemble():
    """Create a simpler ensemble for faster training/testing"""
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=1,
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=1,
    )

    # Simple ensemble with just XGBoost and Random Forest
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf', rf)
        ],
        voting='soft',
        weights=[1, 1],
        n_jobs=1,
    )

    return ensemble
