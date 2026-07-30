# optimize_model.py
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier
import numpy as np

def optimize_xgboost(X_train, y_train):
    """Find best hyperparameters for XGBoost"""

    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }

    xgb = XGBClassifier(random_state=42)

    # Randomized search (faster than grid search) - REDUCED for Streamlit performance
    random_search = RandomizedSearchCV(
        xgb,
        param_distributions=param_grid,
        n_iter=10,  # Reduced from 50 to 10 for faster startup
        scoring='accuracy',
        cv=3,  # Reduced from 5 to 3 folds
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.3f}")

    return random_search.best_estimator_

# Usage
# best_model = optimize_xgboost(X_train, y_train)