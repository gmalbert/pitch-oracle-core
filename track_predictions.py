import pandas as pd
from datetime import datetime
from os import path

import os
DATA_DIR = os.getenv('PITCH_ORACLE_DATA_DIR', 'data_files/')
PREDICTIONS_LOG = path.join(DATA_DIR, 'predictions_log.csv')

def log_prediction(date, home_team, away_team, pred_home, pred_draw, pred_away):
    """Log a prediction for future validation"""
    prediction = {
        'PredictionDate': datetime.now().strftime('%Y-%m-%d'),
        'MatchDate': date,
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'PredHomeWin': pred_home,
        'PredDraw': pred_draw,
        'PredAwayWin': pred_away,
        'ActualResult': None,  # To be filled after match
        'Correct': None
    }

    if path.exists(PREDICTIONS_LOG):
        df = pd.read_csv(PREDICTIONS_LOG)
        df = pd.concat([df, pd.DataFrame([prediction])], ignore_index=True)
    else:
        df = pd.DataFrame([prediction])

    df.to_csv(PREDICTIONS_LOG, index=False)

def validate_predictions():
    """Compare predictions with actual results"""
    if not path.exists(PREDICTIONS_LOG):
        return None

    predictions = pd.read_csv(PREDICTIONS_LOG)
    historical = pd.read_csv(path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv'), sep='\t')

    for idx, pred in predictions.iterrows():
        if pd.isna(pred['ActualResult']):
            # Find the actual match result
            match = historical[
                (historical['MatchDate'] == pred['MatchDate']) &
                (historical['HomeTeam'] == pred['HomeTeam']) &
                (historical['AwayTeam'] == pred['AwayTeam'])
            ]

            if len(match) > 0:
                actual = match.iloc[0]['FullTimeResult']
                predicted = max(
                    [(pred['PredHomeWin'], 'H'),
                     (pred['PredDraw'], 'D'),
                     (pred['PredAwayWin'], 'A')]
                )[1]

                predictions.at[idx, 'ActualResult'] = actual
                predictions.at[idx, 'Correct'] = (predicted == actual)

    predictions.to_csv(PREDICTIONS_LOG, index=False)
    return predictions
