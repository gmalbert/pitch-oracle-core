import pandas as pd
from datetime import datetime
from os import path

import os
DATA_DIR = os.getenv('PITCH_ORACLE_DATA_DIR', 'data_files/')
PREDICTIONS_LOG = path.join(DATA_DIR, 'predictions_log.csv')

# Closing-odds column candidates in the historical data (football-data.co.uk).
_CLOSING_HOME_COLS = ("AvgCH", "Avg_HomeWinOdds", "Pinnacle_ClosingHomeOdds", "Bet365_ClosingHomeOdds")
_CLOSING_DRAW_COLS = ("AvgCD", "Avg_DrawOdds", "Pinnacle_ClosingDrawOdds", "Bet365_ClosingDrawOdds")
_CLOSING_AWAY_COLS = ("AvgCA", "Avg_AwayWinOdds", "Pinnacle_ClosingAwayOdds", "Bet365_ClosingAwayOdds")


def _safe_float(value) -> float | None:
    try:
        v = float(value)
        return v if v > 1 else None
    except (TypeError, ValueError):
        return None


def _closing_odds(row: pd.Series) -> tuple[float, float, float] | None:
    """Extract closing decimal odds from a historical row, if available."""
    home = draw = away = None
    for col in _CLOSING_HOME_COLS:
        home = _safe_float(row.get(col))
        if home is not None:
            break
    for col in _CLOSING_DRAW_COLS:
        draw = _safe_float(row.get(col))
        if draw is not None:
            break
    for col in _CLOSING_AWAY_COLS:
        away = _safe_float(row.get(col))
        if away is not None:
            break
    if home and draw and away:
        return home, draw, away
    return None


def _implied_from_odds(odds: tuple[float, float, float]) -> dict[str, float]:
    """Compute de-vigged market probabilities via penaltyblog.implied (LOGARITHMIC)."""
    from pitch_oracle_core.implied import implied_probabilities
    fair = implied_probabilities(list(odds))
    return {
        "MarketHomeWin": round(fair["home"], 6),
        "MarketDraw": round(fair["draw"], 6),
        "MarketAwayWin": round(fair["away"], 6),
        "MarketMargin": round(fair.margin, 6),
        "MarketMethod": fair.method.value,
    }


def _proper_scores(pred_home: float, pred_draw: float, pred_away: float,
                   actual: str) -> dict[str, float]:
    """Compute per-fixture proper scores via penaltyblog.metrics."""
    import numpy as np
    from pitch_oracle_core.evaluation.baseline import proper_score_summary
    outcome_map = {"H": 0, "D": 1, "A": 2}
    if actual not in outcome_map:
        return {}
    y = np.array([outcome_map[actual]])
    p = np.array([[pred_home, pred_draw, pred_away]])
    scores = proper_score_summary(y, p)
    return {
        "Brier": round(scores["brier"], 6),
        "LogLoss": round(scores["log_loss"], 6),
        "RPS": round(scores["rps"], 6),
    }


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
    """Compare predictions with actual results.

    When closing odds are available in the historical data, de-vigged market
    probabilities and per-fixture proper scores are attached to each resolved
    prediction via ``penaltyblog.implied`` (LOGARITHMIC) and
    ``penaltyblog.metrics``.
    """
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

                # Attach market-implied probabilities from closing odds
                odds = _closing_odds(match.iloc[0])
                if odds:
                    implied = _implied_from_odds(odds)
                    for col, value in implied.items():
                        predictions.at[idx, col] = value

                # Attach per-fixture proper scores
                scores = _proper_scores(
                    pred['PredHomeWin'], pred['PredDraw'], pred['PredAwayWin'], actual,
                )
                for col, value in scores.items():
                    predictions.at[idx, col] = value

    predictions.to_csv(PREDICTIONS_LOG, index=False)
    return predictions
