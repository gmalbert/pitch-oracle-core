"""
scripts/export_best_bets.py - EPL (premier-league)
Reads data_files/predictions_log.csv + data_files/raw/odds.csv (if available),
computes edge, and writes data_files/best_bets_today.json.
"""
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

SPORT = "EPL"
MODEL_VERSION = "1.0.0"
SEASON = str(date.today().year)
OUT_PATH = Path("data_files/best_bets_today.json")
PREDS_PATH = Path("data_files/predictions_log.csv")
ODDS_PATH  = Path("data_files/raw/odds.csv")
MIN_PROBABILITY_EDGE = 0.03
MIN_EXPECTED_VALUE = 0.03


def _write(bets: list, notes: str = "") -> None:
    payload: dict = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "season": SEASON,
        },
        "bets": bets,
    }
    if notes:
        payload["meta"]["notes"] = notes
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[{SPORT}] Wrote {len(bets)} bets -> {OUT_PATH}")


def _tier_from_expected_value(expected_value: float) -> str:
    if expected_value >= 0.12:
        return "Elite"
    elif expected_value >= 0.07:
        return "Strong"
    elif expected_value >= MIN_EXPECTED_VALUE:
        return "Good"
    return "Standard"


def _market_metrics(model_probability: float, decimal_odds: float, all_decimal_odds: list[float]) -> tuple[float, float, float]:
    """Return no-vig market probability, probability edge, and expected return."""
    implied = [1.0 / price for price in all_decimal_odds]
    market_probability = (1.0 / decimal_odds) / sum(implied)
    probability_edge = model_probability - market_probability
    expected_value = model_probability * decimal_odds - 1.0
    return market_probability, probability_edge, expected_value


def _decimal_to_american(dec) -> int | None:
    try:
        dec = float(dec)
        if dec >= 2.0:
            return round((dec - 1) * 100)
        return round(-100 / (dec - 1))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def main() -> None:
    today = date.today()

    if not PREDS_PATH.exists():
        _write([], "No predictions_log.csv; run automation/generate_predictions.py first")
        return

    preds = pd.read_csv(PREDS_PATH)

    # Normalise date column
    date_col = "MatchDate" if "MatchDate" in preds.columns else "Date"
    if date_col not in preds.columns:
        _write([], "No date column in predictions log")
        return

    preds[date_col] = pd.to_datetime(preds[date_col], errors="coerce").dt.date
    today_preds = preds[preds[date_col] == today].copy()

    if today_preds.empty:
        _write([], f"No {SPORT} predictions for {today}")
        return

    # Try to join odds
    odds_available = ODDS_PATH.exists()
    if odds_available:
        try:
            odds = pd.read_csv(ODDS_PATH)
            merge_cols = [c for c in ["HomeTeam", "AwayTeam", date_col] if c in odds.columns]
            if {"HomeTeam", "AwayTeam"}.issubset(merge_cols):
                if date_col in odds.columns:
                    odds[date_col] = pd.to_datetime(odds[date_col], errors="coerce").dt.date
                today_preds = today_preds.merge(odds, on=merge_cols, how="left", suffixes=("", "_odds"))
        except Exception:
            odds_available = False

    outcome_cols = [
        ("Home Win",  "PredHomeWin",  "B365H", "OddsHome"),
        ("Draw",      "PredDraw",     "B365D", "OddsDraw"),
        ("Away Win",  "PredAwayWin",  "B365A", "OddsAway"),
    ]

    bets = []
    for _, row in today_preds.iterrows():
        home = str(row.get("HomeTeam", ""))
        away = str(row.get("AwayTeam", ""))
        game = f"{away} @ {home}"

        model_probabilities = [_safe_float(row.get(item[1])) for item in outcome_cols]
        if any(value is None or value < 0 for value in model_probabilities):
            continue
        probability_total = sum(model_probabilities)
        if probability_total <= 0:
            continue
        model_probabilities = [value / probability_total for value in model_probabilities]

        decimal_odds = []
        for _, _, odds_col1, odds_col2 in outcome_cols:
            price = next(
                (value for column in (odds_col1, odds_col2)
                 if (value := _safe_float(row.get(column))) is not None and value > 1),
                None,
            )
            decimal_odds.append(price)
        if any(price is None for price in decimal_odds):
            continue

        for (outcome, _, _, _), pred_p, odds_val in zip(outcome_cols, model_probabilities, decimal_odds):
            market_probability, edge, expected_value = _market_metrics(
                pred_p, odds_val, decimal_odds
            )
            if edge < MIN_PROBABILITY_EDGE or expected_value < MIN_EXPECTED_VALUE:
                continue

            tier = _tier_from_expected_value(expected_value)
            american = _decimal_to_american(odds_val)

            bet: dict = {
                "game_date": str(today),
                "game_time": None,
                "game": game,
                "home_team": home,
                "away_team": away,
                "bet_type": "Match Result",
                "pick": outcome,
                "confidence": round(pred_p, 4),
                "edge": round(edge, 4),
                "market_probability": round(market_probability, 4),
                "expected_value": round(expected_value, 4),
                "tier": tier,
                "odds": american,
                "line": None,
                "league": "Premier League",
            }
            bets.append(bet)

    _write(bets, "" if bets else f"No qualifying {SPORT} picks for {today}")


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
