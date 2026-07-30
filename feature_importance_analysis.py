import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# Load the data
df = pd.read_csv('data_files/combined_historical_data_with_calculations.csv', sep='\t')

print('=== STEP 2: DETAILED FEATURE IMPORTANCE ANALYSIS ===')

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

# Train model
model = XGBClassifier(eval_metric='mlogloss', random_state=42, max_depth=4, n_estimators=100)
model.fit(X_train, y_train)

print('Model trained successfully!')

# Get feature importances from the model
feature_importance = model.feature_importances_
feature_names = X.columns

# Create DataFrame for analysis
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance,
    'importance_pct': feature_importance * 100
})

# Sort by importance
importance_df = importance_df.sort_values('importance', ascending=False)

print('\n🏆 TOP 20 MOST IMPORTANT FEATURES:')
print('=' * 50)
for i in range(min(20, len(importance_df))):
    row = importance_df.iloc[i]
    print(f'{i+1:2d}. {row["feature"]:<40} {row["importance_pct"]:.3f}%')

# Analyze feature categories
print('\n📊 FEATURE CATEGORY ANALYSIS:')
print('=' * 50)

categories = {
    'Betting Odds': ['Bet365', 'BetWin', 'Interwetten', 'WilliamHill', 'VCBet'],
    'Team Form': ['Last5', 'Rolling', 'Form'],
    'Injury': ['Injury'],
    'Head-to-Head': ['H2H', 'Vs'],
    'Match Stats': ['Shots', 'Corners', 'Fouls', 'Cards'],
    'Goals': ['Goals', 'Scored', 'Conceded'],
    'Points': ['Points']
}

category_importance = {}
category_counts = {}

for category, keywords in categories.items():
    mask = importance_df['feature'].str.contains('|'.join(keywords), case=False, regex=True)
    if mask.any():
        cat_importance = importance_df[mask]['importance'].sum()
        cat_count = mask.sum()
        category_importance[category] = cat_importance
        category_counts[category] = cat_count

# Add 'Other' category
total_importance = importance_df['importance'].sum()
other_importance = total_importance - sum(category_importance.values())
other_count = len(importance_df) - sum(category_counts.values())
category_importance['Other'] = other_importance
category_counts['Other'] = other_count

print('Feature categories by total importance:')
for category in sorted(category_importance.keys(), key=lambda x: category_importance[x], reverse=True):
    pct = (category_importance[category] / total_importance) * 100
    count = category_counts[category]
    print(f'  {category:<15} {count:3d} features | {pct:.1f}% total importance')

# Analyze new features specifically
print('\n🎯 NEW FEATURES DEEP DIVE:')
print('=' * 50)

new_feature_keywords = [
    'Injury', 'ImpliedProb', 'Bet365_MarketMargin', 'OddsMovement',
    'Bet365_Value', 'Bet365_HomeVsDraw', 'Bet365_AwayVsDraw', 'Bet365_HomeVsAway'
]

new_features = []
for keyword in new_feature_keywords:
    mask = importance_df['feature'].str.contains(keyword, case=False)
    if mask.any():
        new_features.extend(importance_df[mask].to_dict('records'))

new_features_df = pd.DataFrame(new_features)
if not new_features_df.empty:
    new_features_df = new_features_df.drop_duplicates(subset='feature')
    new_features_df = new_features_df.sort_values('importance', ascending=False)

    print('New features ranked by importance:')
    for i in range(min(15, len(new_features_df))):
        row = new_features_df.iloc[i]
        print(f'  {row["feature"]:<35} {row["importance_pct"]:.3f}%')

    avg_imp = new_features_df['importance_pct'].mean()
    max_imp = new_features_df['importance_pct'].max()
    min_imp = new_features_df['importance_pct'].min()

    print(f'\nNew features summary:')
    print(f'  Total new features: {len(new_features_df)}')
    print(f'  Average importance: {avg_imp:.3f}%')
    print(f'  Max importance: {max_imp:.3f}%')
    print(f'  Min importance: {min_imp:.3f}%')

# Permutation importance for validation
print('\n🔄 PERMUTATION IMPORTANCE VALIDATION:')
print('=' * 50)

result = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=42, scoring='accuracy')

perm_importance_df = pd.DataFrame({
    'feature': feature_names,
    'perm_importance': result.importances_mean,
    'perm_importance_pct': result.importances_mean * 100
})

perm_importance_df = perm_importance_df.sort_values('perm_importance', ascending=False)

print('Top 10 features by permutation importance:')
for i in range(min(10, len(perm_importance_df))):
    row = perm_importance_df.iloc[i]
    print(f'  {row["feature"]:<35} {row["perm_importance_pct"]:.3f}%')