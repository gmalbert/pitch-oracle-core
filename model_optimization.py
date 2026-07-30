import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

# Load the data
df = pd.read_csv('data_files/combined_historical_data_with_calculations.csv', sep='\t')

print('=== STEP 3: HYPERPARAMETER OPTIMIZATION ===')

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

# Handle categorical columns properly
for col in X.select_dtypes(include='object').columns:
    if X[col].nunique() < 50:
        X[col] = X[col].astype('category').cat.codes
    else:
        X = X.drop(columns=[col])

# Fill NaN values with column means for numeric columns only
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

# Ensure all columns are numeric
X = X.select_dtypes(include=[np.number])

# Replace any remaining NaN/inf with 0
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Clean column names for XGBoost
X.columns = [str(col).replace('[','').replace(']','').replace('<','').replace('>','').replace(' ', '_') for col in X.columns]

# Convert to numpy arrays explicitly
X_np = X.values.astype(np.float32)
y_np = y.values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42, stratify=y_np)

print('Data prepared successfully!')
print(f'Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}')
print(f'Features: {X_train.shape[1]}')

# Baseline model (current best)
print('\n📊 BASELINE MODEL PERFORMANCE:')
baseline_model = XGBClassifier(eval_metric='mlogloss', random_state=42, max_depth=4, n_estimators=100)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_pred)
baseline_mae = mean_absolute_error(y_test, baseline_pred)
print(f'  Accuracy: {baseline_acc:.3f}')
print(f'  MAE: {baseline_mae:.3f}')

# Hyperparameter tuning
print('\n🔧 HYPERPARAMETER OPTIMIZATION:')

# Define parameter grid
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

# Create base model
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)

# Grid search with cross-validation
print('Running grid search (this may take a few minutes)...')
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters and score
print(f'\n🏆 BEST PARAMETERS FOUND:')
best_params = grid_search.best_params_
for param, value in best_params.items():
    print(f'  {param}: {value}')

print(f'\nBest cross-validation accuracy: {grid_search.best_score_:.3f}')

# Train optimized model
print('\n🚀 TRAINING OPTIMIZED MODEL:')
optimized_model = XGBClassifier(**best_params, eval_metric='mlogloss', random_state=42)
optimized_model.fit(X_train, y_train)

# Evaluate optimized model
opt_pred = optimized_model.predict(X_test)
opt_acc = accuracy_score(y_test, opt_pred)
opt_mae = mean_absolute_error(y_test, opt_pred)

print(f'\n📈 OPTIMIZED MODEL PERFORMANCE:')
print(f'  Accuracy: {opt_acc:.3f} (vs baseline: {baseline_acc:.3f})')
print(f'  MAE: {opt_mae:.3f} (vs baseline: {baseline_mae:.3f})')
print(f'  Improvement: +{(opt_acc - baseline_acc)*100:.2f}% accuracy')

# Detailed classification report
print(f'\n📋 CLASSIFICATION REPORT:')
target_names = ['Home Win', 'Draw', 'Away Win']
print(classification_report(y_test, opt_pred, target_names=target_names))

# Feature importance of optimized model
print(f'\n🔍 TOP 10 FEATURES (OPTIMIZED MODEL):')
opt_importance = optimized_model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': opt_importance,
    'importance_pct': opt_importance * 100
}).sort_values('importance', ascending=False)

for i in range(min(10, len(importance_df))):
    row = importance_df.iloc[i]
    print(f'  {row["feature"]:<35} {row["importance_pct"]:.3f}%')

# Compare with baseline feature importance
baseline_importance = baseline_model.feature_importances_
importance_change = opt_importance - baseline_importance
importance_change_df = pd.DataFrame({
    'feature': feature_names,
    'change': importance_change,
    'change_pct': importance_change * 100
}).sort_values('change', ascending=False)

print(f'\n⚡ FEATURE IMPORTANCE CHANGES (Top 5 increases):')
for i in range(min(5, len(importance_change_df))):
    row = importance_change_df.iloc[i]
    if row['change'] > 0:
        print(f'  +{row["change_pct"]:+.3f}% {row["feature"]}')

print(f'\n📊 OPTIMIZATION SUMMARY:')
print(f'  Baseline accuracy: {baseline_acc:.3f}')
print(f'  Optimized accuracy: {opt_acc:.3f}')
print(f'  Net improvement: +{(opt_acc - baseline_acc)*100:.2f}%')
print(f'  Best parameters: max_depth={best_params["max_depth"]}, n_estimators={best_params["n_estimators"]}, learning_rate={best_params["learning_rate"]}')