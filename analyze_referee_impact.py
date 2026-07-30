import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from xgboost import XGBClassifier
import os

# Load the data with referee statistics
DATA_DIR = 'data_files/'
csv_path = os.path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv')
df = pd.read_csv(csv_path, sep='\t')

print('Dataset shape:', df.shape)
print('Matches:', len(df))
print()

# Data preparation
target_map = {'H': 0, 'D': 1, 'A': 2}
df = df[df['FullTimeResult'].isin(target_map.keys())].copy()
df['target'] = df['FullTimeResult'].map(target_map)

# Drop columns not useful for modeling
drop_cols = [
    'MatchDate', 'KickoffTime', 'FullTimeResult', 'HomeTeam', 'AwayTeam', 'WinningTeam',
    'HomeWin', 'AwayWin', 'Draw', 'HalfTimeHomeWin', 'HalfTimeAwayWin', 'HalfTimeDraw',
    'FullTimeHomeGoals', 'FullTimeAwayGoals', 'HalfTimeResult', 'HalfTimeHomeGoals', 'HalfTimeAwayGoals',
    'HomePoints', 'AwayPoints', 'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget',
    'HomeFouls', 'AwayFouls', 'HomeCorners', 'AwayCorners', 'HomeYellowCards', 'AwayYellowCards',
    'HomeRedCards', 'AwayRedCards'
]
X = df.drop(columns=[col for col in drop_cols if col in df.columns] + ['target'], errors='ignore')
y = df['target']

print('Features before processing:', X.shape[1])

# Fill NA and encode categoricals
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category').cat.codes
X = X.fillna(X.mean(numeric_only=True))

# Select only numeric columns
X = X.select_dtypes(include=[np.number])

# Clean column names
X.columns = [str(col).replace('[','').replace(']','').replace('<','').replace('>','').replace(' ', '_') for col in X.columns]

print(f"Final features after cleanup: {X.shape[1]}")

print('Final features:', X.shape[1])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# XGBoost Model
model = XGBClassifier(eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Predictions & Metrics
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print()
print('MODEL PERFORMANCE WITH REFEREE FEATURES:')
print(f'Accuracy: {acc:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print()

# Get feature importance from the model
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance,
    'Importance %': feature_importance * 100
})
importance_df = importance_df.sort_values('Importance', ascending=False)

print('TOP 15 MOST IMPORTANT FEATURES:')
for idx, row in importance_df.head(15).iterrows():
    feature_name = str(row['Feature'])[:34]  # Truncate long names
    print(f'{feature_name:<35} {row["Importance %"]:>8.2f}%')

print()
print('REFEREE FEATURE IMPORTANCE:')
referee_features = [col for col in importance_df['Feature'] if col.startswith('Ref')]
referee_importance = importance_df[importance_df['Feature'].isin(referee_features)]
referee_importance = referee_importance.sort_values('Importance', ascending=False)

for idx, row in referee_importance.iterrows():
    feature_name = str(row['Feature'])[:34]  # Truncate long names
    print(f'{feature_name:<35} {row["Importance %"]:>8.2f}%')

print()
print('SUMMARY:')
print(f'• Total features: {X.shape[1]} (increased by 8 referee features)')
print(f'• Referee features represent: {len(referee_features)/X.shape[1]*100:.1f}% of total features')
if len(referee_importance) > 0:
    top_ref = referee_importance.iloc[0]
    print(f'• Top referee feature: {str(top_ref["Feature"])[:30]} ({top_ref["Importance %"]:.2f}%)')