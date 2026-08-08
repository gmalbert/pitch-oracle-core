"""
Pre-train ML models for a Pitch Oracle consumer
This script is run by the automated nightly pipeline to pre-train models
so they're ready for the next day, improving app startup performance.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import time
import gc
from os import path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from models.no_odds_predictor import create_no_odds_classifier
from models.neural_predictor import train_neural_model, predict_neural
from models.poisson_evaluation import evaluate_poisson_file
from models.lstm_predictor import train_lstm_model
from optimize_model import optimize_xgboost
from pitch_oracle_core.features import (
    FEATURE_POLICY_VERSION,
    chronological_partition_indices,
    completed_future_rows,
    no_odds_feature_columns,
)
from pitch_oracle_core.model_audit import (
    TemperatureScaledClassifier,
    fit_temperature,
    probability_metrics,
)

DATA_DIR = os.getenv('PITCH_ORACLE_DATA_DIR', 'data_files/')
MODELS_DIR = os.getenv('PITCH_ORACLE_MODELS_DIR', 'models/')


def _normalized_probabilities(values):
    probabilities = np.asarray(values, dtype=float)
    totals = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (totals <= 0).any():
        raise ValueError("Model emitted invalid class probabilities")
    return probabilities / totals

def load_and_preprocess_data():
    """Load processed data and prepare for training"""
    csv_path = path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv')

    if not path.exists(csv_path):
        raise FileNotFoundError(f"Processed data file not found: {csv_path}")

    df = pd.read_csv(csv_path, sep='\t')
    invalid_future = completed_future_rows(df)
    if not invalid_future.empty:
        examples = invalid_future['MatchDate'].astype(str).head(3).tolist()
        raise ValueError(
            "Historical data contains completed matches dated in the future "
            f"({len(invalid_future)} rows; examples: {examples}). Regenerate raw history "
            "with explicit day-first date parsing before training."
        )

    # Target variable (3-class: Home Win=0, Draw=1, Away Win=2)
    target_map = {'H': 0, 'D': 1, 'A': 2}
    valid = df['FullTimeResult'].isin(target_map) & pd.to_datetime(df['MatchDate'], errors='coerce').notna()
    df = df.loc[valid].copy()
    df['_model_date'] = pd.to_datetime(df['MatchDate'], errors='raise')
    df = df.sort_values('_model_date', kind='stable').reset_index(drop=True)
    y = df['FullTimeResult'].map(target_map)
    dates = df['_model_date']
    feature_frame = df[no_odds_feature_columns(df)]

    # Get numeric features only
    X_numeric = feature_frame.select_dtypes(include=[np.number])

    # Do not train on categorical values until their fitted encoder is persisted
    # and reused by consumer inference. Live prediction previously filled every
    # categorical feature with zero, which was a train/serve mismatch.
    feature_names = X_numeric.columns.tolist()
    X = X_numeric.copy()
    X.columns = [f'feature_{i}' for i in range(X.shape[1])]
    return X, y.to_numpy(), dates.to_numpy(), feature_names


def _calibrate(estimator, X_calibration, y_calibration):
    temperature = fit_temperature(estimator.predict_proba(X_calibration), y_calibration)
    return TemperatureScaledClassifier(estimator, temperature)

def train_and_save_models():
    """Train all models and save them to disk"""
    start_time = time.time()
    print(f"Starting model training pipeline at {time.strftime('%H:%M:%S')}")
    print(f"Code version: feature contract v2 - {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("Loading and preprocessing data...")
    data_start = time.time()
    X, y, dates, feature_names = load_and_preprocess_data()
    print(f"Data loaded in {time.time() - data_start:.2f}s")

    print("Splitting data...")
    train_indices, calibration_indices, test_indices = chronological_partition_indices(
        dates, calibration_size=0.2, test_size=0.2
    )
    X_train_frame = X.iloc[train_indices].copy()
    X_calibration_frame = X.iloc[calibration_indices].copy()
    X_test_frame = X.iloc[test_indices].copy()
    train_means = X_train_frame.mean().fillna(0.0)
    X_train = X_train_frame.fillna(train_means).fillna(0.0).to_numpy()
    X_calibration = X_calibration_frame.fillna(train_means).fillna(0.0).to_numpy()
    X_test = X_test_frame.fillna(train_means).fillna(0.0).to_numpy()
    y_train, y_calibration, y_test = (
        y[train_indices], y[calibration_indices], y[test_indices]
    )

    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Train XGBoost baseline
    print("Training XGBoost baseline...")
    xgb_start = time.time()
    xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=1)
    xgb_model.fit(X_train, y_train)
    xgb_model = _calibrate(xgb_model, X_calibration, y_calibration)

    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_log_loss = log_loss(
        y_test, _normalized_probabilities(xgb_model.predict_proba(X_test)), labels=[0, 1, 2]
    )

    print(f"XGBoost accuracy: {xgb_acc:.3f}")
    print(f"XGBoost trained in {time.time() - xgb_start:.2f}s")

    # Save XGBoost model
    with open(path.join(MODELS_DIR, 'xgb_baseline.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)

    # 2. Train the production no-odds probability model. Rolling-origin
    # ablation selected regularized multinomial logistic regression because it
    # beat both the class-prior baseline and the higher-variance tree ensemble
    # on log loss, Brier score and calibration.
    print("Training production no-odds model...")
    ensemble_start = time.time()
    ensemble_model = create_no_odds_classifier()
    ensemble_model.fit(X_train, y_train)
    ensemble_model = _calibrate(ensemble_model, X_calibration, y_calibration)

    ensemble_pred = ensemble_model.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_log_loss = log_loss(
        y_test, _normalized_probabilities(ensemble_model.predict_proba(X_test)), labels=[0, 1, 2]
    )

    print(f"Production no-odds accuracy: {ensemble_acc:.3f}")
    print(f"Production no-odds model trained in {time.time() - ensemble_start:.2f}s")

    train_prior = np.bincount(y_train, minlength=3).astype(float)
    train_prior /= train_prior.sum()
    baseline_probabilities = np.tile(train_prior, (len(y_test), 1))
    baseline_metrics = probability_metrics(y_test, baseline_probabilities)
    production_metrics = probability_metrics(y_test, ensemble_model.predict_proba(X_test))
    if (
        production_metrics['log_loss'] >= baseline_metrics['log_loss']
        or production_metrics['brier_score'] >= baseline_metrics['brier_score']
    ):
        raise RuntimeError(
            "Production no-odds model failed the release gate: "
            f"model log_loss={production_metrics['log_loss']:.4f}, "
            f"baseline={baseline_metrics['log_loss']:.4f}; "
            f"model brier={production_metrics['brier_score']:.4f}, "
            f"baseline={baseline_metrics['brier_score']:.4f}"
        )

    # Save Ensemble model
    with open(path.join(MODELS_DIR, 'ensemble_model.pkl'), 'wb') as f:
        pickle.dump(ensemble_model, f)

    # 3. Train Neural Network
    print("Training Neural Network (this may take several minutes)...")
    neural_start = time.time()
    try:
        neural_model, neural_scaler = train_neural_model(X_train, y_train, epochs=50, batch_size=32)

        neural_pred_proba = _normalized_probabilities(
            predict_neural(neural_model, neural_scaler, X_test)
        )
        neural_pred = np.argmax(neural_pred_proba, axis=1)
        neural_acc = accuracy_score(y_test, neural_pred)
        neural_log_loss = log_loss(y_test, neural_pred_proba, labels=[0, 1, 2])

        print(f"Neural-network accuracy: {neural_acc:.3f}")
        print(f"Neural network trained in {time.time() - neural_start:.2f}s")

        # Save Neural Network model and scaler
        with open(path.join(MODELS_DIR, 'neural_model.pkl'), 'wb') as f:
            pickle.dump(neural_model, f)
        with open(path.join(MODELS_DIR, 'neural_scaler.pkl'), 'wb') as f:
            pickle.dump(neural_scaler, f)
            
    except Exception as e:
        print(f"WARNING: Neural Network training failed: {e}")
        print("Continuing without neural network model...")
        neural_model = None
        neural_scaler = None
        neural_acc = neural_log_loss = 0

    # 4. Train Optimized XGBoost
    print("Training Optimized XGBoost (hyperparameter search)...")
    opt_start = time.time()
    optimized_xgb_model = optimize_xgboost(X_train, y_train)
    optimized_xgb_model = _calibrate(
        optimized_xgb_model, X_calibration, y_calibration
    )

    opt_xgb_pred = optimized_xgb_model.predict(X_test)
    opt_xgb_acc = accuracy_score(y_test, opt_xgb_pred)
    opt_xgb_log_loss = log_loss(
        y_test,
        _normalized_probabilities(optimized_xgb_model.predict_proba(X_test)),
        labels=[0, 1, 2],
    )

    print(f"Optimized XGBoost accuracy: {opt_xgb_acc:.3f}")
    print(f"Optimized XGBoost trained in {time.time() - opt_start:.2f}s")

    # Save Optimized XGBoost model
    with open(path.join(MODELS_DIR, 'optimized_xgb.pkl'), 'wb') as f:
        pickle.dump(optimized_xgb_model, f)

    # Save model performance metrics
    performance = {
        'class_prior_baseline': baseline_metrics,
        'xgb_baseline': probability_metrics(y_test, xgb_model.predict_proba(X_test)),
        'ensemble': production_metrics,
        'optimized_xgb': probability_metrics(
            y_test, optimized_xgb_model.predict_proba(X_test)
        ),
    }
    
    # Only include neural network if it was successfully trained
    if neural_model is not None:
        performance['neural'] = {'accuracy': neural_acc, 'log_loss': neural_log_loss}

    # Model fitting libraries retain sizeable native buffers. Release fitted
    # estimators after serialization so the subsequent LSTM does not inherit
    # the peak memory footprint of the entire training pipeline.
    del xgb_model, ensemble_model, optimized_xgb_model
    if neural_model is not None:
        del neural_model, neural_scaler
    gc.collect()

    # evaluate poisson metrics on the same historical data
    try:
        poisson_metrics = evaluate_poisson_file(path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv'))
        performance['poisson'] = poisson_metrics
        # append to history CSV
        hist_path = path.join(DATA_DIR, 'poisson_metrics_history.csv')
        hist_df = pd.DataFrame([{
            'date': pd.Timestamp.now(),
            'home_mae': poisson_metrics['home_mae'],
            'away_mae': poisson_metrics['away_mae'],
            'home_rmse': poisson_metrics['home_rmse'],
            'away_rmse': poisson_metrics['away_rmse'],
            'outcome_acc': poisson_metrics['outcome_acc']
        }])
        if path.exists(hist_path):
            hist_df.to_csv(hist_path, mode='a', header=False, index=False)
        else:
            hist_df.to_csv(hist_path, index=False)
    except Exception as e:
        print(f"WARNING: Poisson evaluation failed: {e}")

    # Train and save LSTM model
    try:
        print("\nTraining LSTM time series model...")
        lstm_start = time.time()
        raw_df = pd.read_csv(path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv'), sep='\t')
        lstm_predictor = train_lstm_model(raw_df, sequence_length=5, epochs=30)
        lstm_predictor.save_model(path.join(MODELS_DIR, 'lstm_predictor.pkl'))
        print(f"LSTM model trained and saved in {time.time() - lstm_start:.1f}s")
    except Exception as e:
        print(f"WARNING: LSTM training failed: {e}")

    with open(path.join(MODELS_DIR, 'model_performance.pkl'), 'wb') as f:
        pickle.dump(performance, f)

    metadata = {
        'feature_policy_version': FEATURE_POLICY_VERSION,
        'feature_set': 'no_odds',
        'feature_names': feature_names,
        'feature_count': len(feature_names),
        'split_strategy': 'chronological_train_calibration_test',
        'train_rows': len(train_indices),
        'calibration_rows': len(calibration_indices),
        'test_rows': len(test_indices),
        'calibration_method': 'temperature_scaling',
        'production_model': 'regularized_multinomial_logistic_regression',
        'release_gate': 'beats_training_class_prior_on_log_loss_and_brier',
        'generated_at': pd.Timestamp.now(tz='UTC').isoformat(),
    }
    with open(path.join(MODELS_DIR, 'model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - start_time
    print("All models trained and saved successfully!")
    print(f"Total training time: {total_time:.2f}s")
    print("\nModel Performance Summary:")
    for model_name, metrics in performance.items():
        if model_name == 'poisson':
            # poisson entry contains detailed metrics
            print(f"poisson: home_mae={metrics['home_mae']:.3f}, away_mae={metrics['away_mae']:.3f}, outcome_acc={metrics['outcome_acc']:.3f}")
        else:
            print(f"{model_name}: Accuracy={metrics['accuracy']:.3f}, Log loss={metrics['log_loss']:.3f}")
    return performance

if __name__ == "__main__":
    train_and_save_models()
