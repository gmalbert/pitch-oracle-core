"""
Precompute Database Script

This script processes the raw historical data and saves it in an optimized format
for fast loading in the Streamlit app. This eliminates the expensive CSV parsing
and feature engineering that happens at app startup.

Run this script:
- Locally: python precompute_database.py
- Automated: Via GitHub Actions after data updates
"""

import pandas as pd
import numpy as np
import pickle
import os
from os import path
import warnings
import time
from pitch_oracle_core.features import (
    FEATURE_POLICY_VERSION,
    chronological_partition_indices,
    completed_future_rows,
    no_odds_feature_columns,
)

warnings.filterwarnings('ignore')

DATA_DIR = os.getenv('PITCH_ORACLE_DATA_DIR', 'data_files/')
OUTPUT_DIR = 'precomputed/'

def precompute_data():
    """
    Precompute expensive data processing operations for fast app loading.
    
    This function:
    1. Loads the raw CSV data
    2. Performs feature engineering and encoding
    3. Creates train/test splits
    4. Saves processed data to pickle for instant loading
    
    Expected speedup: 6-10x faster app startup (from 30-60s to 5-10s)
    """
    start_time = time.time()
    print("Starting data precomputation...")
    
    # Load raw data
    csv_path = path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv')
    
    if not path.exists(csv_path):
        print(f"ERROR: Data file not found at {csv_path}")
        return False
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, sep='\t')
    invalid_future = completed_future_rows(df)
    if not invalid_future.empty:
        examples = invalid_future['MatchDate'].astype(str).head(3).tolist()
        raise ValueError(
            "Historical data contains completed matches dated in the future "
            f"({len(invalid_future)} rows; examples: {examples}). Regenerate raw history "
            "before precomputing model artifacts."
        )
    initial_rows = len(df)
    print(f"   Loaded {initial_rows:,} rows")
    
    # Data preparation (matches app logic exactly)
    print("Processing features...")
    target_map = {'H': 0, 'D': 1, 'A': 2}
    df = df[df['FullTimeResult'].isin(target_map.keys())].copy()
    df['target'] = df['FullTimeResult'].map(target_map)
    dates = pd.to_datetime(df['MatchDate'], errors='coerce')
    valid_dates = dates.notna()
    df, dates = df.loc[valid_dates].copy(), dates.loc[valid_dates]
    X = df[no_odds_feature_columns(df)]
    y = df['target']
    
    # Process numeric features
    X_numeric = X.select_dtypes(include=[np.number])
    print(f"   Found {len(X_numeric.columns)} numeric features")
    
    # Categorical encoders are intentionally omitted until the fitted encoder
    # can be persisted with the model and reused identically at inference.
    X = X_numeric.copy()
    
    # Ensure consistent feature names
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X.columns = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Create train/test split (consistent with app)
    print("Creating train/test split...")
    train_indices, calibration_indices, test_indices = chronological_partition_indices(
        dates, calibration_size=0.2, test_size=0.2
    )
    train_means = X.iloc[train_indices].mean().fillna(0.0)
    imputation_values = {
        source_name: float(train_means.iloc[position])
        for position, source_name in enumerate(feature_names)
    }
    # Persist the perspective mapping once with the training contract. Runtime
    # lookup consumes this explicit mapping and never guesses from feature names.
    state_sources = {}
    feature_set = set(feature_names)
    for feature in feature_names:
        if feature.startswith("Home"):
            counterpart = "Away" + feature[len("Home"):]
            fixture_role = "home"
            home_column, away_column = feature, counterpart
        elif feature.startswith("Away"):
            counterpart = "Home" + feature[len("Away"):]
            fixture_role = "away"
            home_column, away_column = counterpart, feature
        else:
            continue
        if counterpart in feature_set:
            state_sources[feature] = {
                "fixture_role": fixture_role,
                "home_history_column": home_column,
                "away_history_column": away_column,
            }
    X = X.fillna(train_means).fillna(0.0)
    X_processed = X.values
    y_processed = y.values
    X_train, X_test = X_processed[train_indices], X_processed[test_indices]
    X_calibration = X_processed[calibration_indices]
    y_train, y_test = y_processed[train_indices], y_processed[test_indices]
    y_calibration = y_processed[calibration_indices]
    
    # Package data for saving
    preprocessed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'X_calibration': X_calibration,
        'y_train': y_train,
        'y_test': y_test,
        'y_calibration': y_calibration,
        'feature_names': feature_names,
        'feature_contract': {
            'version': FEATURE_POLICY_VERSION,
            'feature_names': feature_names,
            'imputation_values': imputation_values,
            'state_sources': state_sources,
        },
        'df_sample': df.head(1000),  # Small sample for quick operations
        'metadata': {
            'total_samples': len(X_processed),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'calibration_samples': len(X_calibration),
            'num_features': X_processed.shape[1],
            'processed_date': pd.Timestamp.now().isoformat(),
            'source_file': csv_path,
            'feature_policy_version': FEATURE_POLICY_VERSION,
            'feature_set': 'no_odds',
            'split_strategy': 'chronological_train_calibration_test',
        }
    }
    
    # Save to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = path.join(OUTPUT_DIR, 'preprocessed_data.pkl')
    
    print(f"Saving preprocessed data to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    # Calculate file sizes
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*60)
    print("PRECOMPUTATION COMPLETE")
    print("="*60)
    print("Summary:")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Total features: {X_processed.shape[1]}")
    print(f"   Output file: {output_path}")
    print(f"   File size: {file_size:.2f} MB")
    print(f"   Processing time: {elapsed_time:.2f} seconds")
    print("\nExpected app startup speedup: 6-10x faster")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = precompute_data()
    exit(0 if success else 1)
