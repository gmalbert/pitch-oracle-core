"""Export the daily best-bets artifact for any Pitch Oracle league."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from .leagues import get_league_config

MIN_EDGE = 0.03
MIN_EXPECTED_VALUE = 0.03


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _decimal_to_american(decimal_odds: object) -> int | None:
    odds = _safe_float(decimal_odds)
    if odds is None or odds <= 1:
        return None
    return round((odds - 1) * 100) if odds >= 2 else round(-100 / (odds - 1))


def market_metrics(model_probability: float, decimal_odds: float,
                   all_decimal_odds: list[float]) -> tuple[float, float, float]:
    """Return no-vig market probability, probability edge, and expected value."""
    implied = [1.0 / odds for odds in all_decimal_odds]
    market_probability = (1.0 / decimal_odds) / sum(implied)
    edge = model_probability - market_probability
    expected_value = model_probability * decimal_odds - 1.0
    return market_probability, edge, expected_value


def _tier(expected_value: float) -> str:
    if expected_value >= 0.12:
        return "Elite"
    if expected_value >= 0.07:
        return "Strong"
    if expected_value >= MIN_EXPECTED_VALUE:
        return "Good"
    return "Standard"


def _write(path: Path, sport: str, league: str, bets: list[dict], notes: str = "") -> None:
    meta = {
        "sport": sport,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "core-" + _core_version(),
        "season": str(date.today().year),
    }
    if notes:
        meta["notes"] = notes
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "bets": bets}, indent=2, ensure_ascii=False), encoding="utf-8")


def _core_version() -> str:
    from ._version import __version__
    return __version__


def _league_context() -> tuple[str, str]:
    key = os.environ.get("PITCH_ORACLE_LEAGUE", "epl").lower()
    config = get_league_config(key)
    return key, config.display_name


def _load_predictions(data_dir: Path) -> tuple[pd.DataFrame | None, str]:
    for name in ("predictions_log.csv", "upcoming_predictions.csv"):
        path = data_dir / name
        if path.exists():
            return pd.read_csv(path), name
    return None, ""


def generate(data_dir: str | Path = "data_files") -> Path:
    """Generate ``best_bets_today.json`` from predictions and optional odds."""
    data_dir = Path(data_dir)
    output = data_dir / "best_bets_today.json"
    sport, league = _league_context()
    predictions, _ = _load_predictions(data_dir)
    if predictions is None:
        _write(output, sport, league, [], "No predictions artifact; run the prediction pipeline first")
        return output

    date_col = "MatchDate" if "MatchDate" in predictions.columns else "Date"
    if date_col not in predictions.columns:
        _write(output, sport, league, [], "Predictions artifact has no date column")
        return output
    predictions[date_col] = pd.to_datetime(predictions[date_col], errors="coerce").dt.date
    today = date.today()
    # Only keep future matches (or today) that haven't been played yet.
    predictions = predictions[predictions[date_col] >= today].copy()
    if predictions.empty:
        _write(output, sport, league, [], f"No upcoming {league} predictions")
        return output

    odds_path = data_dir / "raw" / "odds.csv"
    if not odds_path.exists():
        odds_path = data_dir / "odds.csv"
    if not odds_path.exists():
        _write(output, sport, league, [], f"No odds artifact; best bets require market prices")
        return output
    odds = pd.read_csv(odds_path)
    merge_cols = [column for column in ("HomeTeam", "AwayTeam", date_col) if column in odds.columns]
    if not {"HomeTeam", "AwayTeam"}.issubset(merge_cols):
        _write(output, sport, league, [], "Odds artifact has no home/away team columns")
        return output
    if date_col in odds.columns:
        odds[date_col] = pd.to_datetime(odds[date_col], errors="coerce").dt.date
    predictions = predictions.merge(odds, on=merge_cols, how="left", suffixes=("", "_odds"))

    outcomes = (("Home Win", "HomeWin_Prob", "PredHomeWin", "B365H", "OddsHome"),
                ("Draw", "Draw_Prob", "PredDraw", "B365D", "OddsDraw"),
                ("Away Win", "AwayWin_Prob", "PredAwayWin", "B365A", "OddsAway"))
    bets: list[dict] = []
    for _, row in predictions.iterrows():
        home, away = str(row.get("HomeTeam", "")), str(row.get("AwayTeam", ""))
        probabilities = []
        for _, modern, legacy, _, _ in outcomes:
            probabilities.append(_safe_float(row.get(modern)) or _safe_float(row.get(legacy)))
        if any(value is None or value < 0 for value in probabilities):
            continue
        total = sum(probabilities)
        probabilities = [value / total for value in probabilities] if total > 0 else []
        odds_values = []
        for _, _, _, primary, fallback in outcomes:
            odds_values.append(next((_safe_float(row.get(column)) for column in (primary, fallback)
                                     if (_safe_float(row.get(column)) or 0) > 1), None))
        if len(probabilities) != 3 or any(value is None for value in odds_values):
            continue
        for (outcome, _, _, _, _), probability, odds_value in zip(outcomes, probabilities, odds_values):
            market_probability, edge, expected_value = market_metrics(probability, odds_value, odds_values)
            if edge < MIN_EDGE or expected_value < MIN_EXPECTED_VALUE:
                continue
            bets.append({
                "game_date": str(row.get(date_col)), "game_time": row.get("Time"),
                "game": f"{away} @ {home}", "home_team": home, "away_team": away,
                "bet_type": "Match Result", "pick": outcome,
                "confidence": round(probability, 4), "edge": round(edge, 4),
                "market_probability": round(market_probability, 4),
                "expected_value": round(expected_value, 4), "tier": _tier(expected_value),
                "odds": _decimal_to_american(odds_value), "line": None, "league": league,
            })
    note = "" if bets else f"No qualifying {league} picks"
    _write(output, sport, league, bets, note)
    return output


if __name__ == "__main__":
    print(f"Wrote {generate(os.environ.get('PITCH_ORACLE_DATA_DIR', 'data_files'))}")
