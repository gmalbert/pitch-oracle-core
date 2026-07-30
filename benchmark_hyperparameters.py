"""
Benchmark Hyperparameter Optimization for Premier League Prediction Models

This script provides a quick way to test and benchmark different hyperparameter
optimization approaches for machine learning models used in Premier League match prediction.

Features:
- Loads and prepares Premier League historical data
- Trains baseline XGBoost model
- Runs hyperparameter optimization using RandomizedSearchCV
- Compares baseline vs optimized model performance
- Prints detailed performance metrics

Usage:
    python benchmark_hyperparameters.py

Output:
- Baseline model performance (accuracy, MAE)
- Optimized model performance (accuracy, MAE)
- Improvement metrics
- Best hyperparameters found
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from xgboost import XGBClassifier
from optimize_model import optimize_xgboost

# Load data
df = pd.read_csv('data_files/combined_historical_data_with_calculations_new.csv', sep='\t')

# Data preparation
target_map = {'H': 0, 'D': 1, 'A': 2}
df = df[df['FullTimeResult'].isin(target_map.keys())].copy()
df['target'] = df['FullTimeResult'].map(target_map)

# Drop columns not useful for modeling
drop_cols = [
    'MatchDate', 'KickoffTime', 'FullTimeResult', 'HomeTeam', 'AwayTeam', 'WinningTeam',
    'HomeWin', 'AwayWin', 'Draw', 'HalfTimeHomeWin', 'HalfTimeAwayWin', 'HalfTimeDraw',
    'FullTimeHomeGoals', 'FullTimeAwayGoals', 'HalfTimeResult', 'HalfTimeHomeGoals',
    'HalfTimeAwayGoals', 'HomePoints', 'AwayPoints'
]

X = df.drop(columns=[col for col in drop_cols if col in df.columns] + ['target'], errors='ignore')
y = df['target']

# Handle categorical columns
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category').cat.codes
X = X.fillna(X.mean(numeric_only=True))

# Select only numeric columns
X = X.select_dtypes(include=[np.number])

# Clean column names for XGBoost compatibility
X.columns = [str(col).replace('[','').replace(']','').replace('<','').replace('>','').replace(' ', '_') for col in X.columns]

# Remove columns that are not 1D or have object dtype
bad_cols = []
for col in X.columns:
    if isinstance(X[col].iloc[0], (pd.Series, pd.DataFrame)) or X[col].dtype == 'O':
        bad_cols.append(col)
if bad_cols:
    X = X.drop(columns=bad_cols)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print('Data prepared successfully!')
print(f'Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}')
print(f'Features: {X_train.shape[1]}')

# Baseline model
print('\n=== BASELINE MODEL ===')
baseline_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_pred)
baseline_mae = mean_absolute_error(y_test, baseline_pred)
print(f'Baseline - Accuracy: {baseline_acc:.3f}, MAE: {baseline_mae:.3f}')

# Optimized model
print('\n=== OPTIMIZED MODEL ===')
best_model = optimize_xgboost(X_train, y_train)
opt_pred = best_model.predict(X_test)
opt_acc = accuracy_score(y_test, opt_pred)
opt_mae = mean_absolute_error(y_test, opt_pred)
print(f'Optimized - Accuracy: {opt_acc:.3f}, MAE: {opt_mae:.3f}')
print(f'Improvement: +{(opt_acc - baseline_acc)*100:.2f}% accuracy, -{(baseline_mae - opt_mae):.3f} MAE')