"""Daily value scan — find value bets and arbitrage opportunities.

Run: python scripts/value/daily_value_scan.py
"""

import os
from pathlib import Path

import pandas as pd

from pitch_oracle_core.betting import find_arbitrage, find_value_bets
from pitch_oracle_core.implied import implied_probabilities


def main():
    data_dir = Path(os.getenv("PITCH_ORACLE_DATA_DIR", "data_files"))
    preds_path = data_dir / "upcoming_predictions.csv"
    odds_path = data_dir / "odds.csv"

    if not preds_path.exists():
        print(f"No predictions at {preds_path}")
        return

    preds = pd.read_csv(preds_path)
    if not odds_path.exists():
        odds_path = data_dir / "raw" / "odds.csv"
    if not odds_path.exists():
        print("No odds artifact; value scan requires market prices")
        return

    odds = pd.read_csv(odds_path)
    merge_cols = [c for c in ("HomeTeam", "AwayTeam", "MatchDate") if c in odds.columns and c in preds.columns]
    if not {"HomeTeam", "AwayTeam"}.issubset(merge_cols):
        print("Odds artifact has no home/away team columns")
        return

    merged = preds.merge(odds, on=merge_cols, how="left", suffixes=("", "_odds"))
    rows = []
    for _, row in merged.iterrows():
        home = str(row.get("HomeTeam", ""))
        away = str(row.get("AwayTeam", ""))
        probs = []
        odds_vals = []
        for prob_col, odds_col in [("HomeWin_Prob", "B365H"), ("Draw_Prob", "B365D"), ("AwayWin_Prob", "B365A")]:
            p = row.get(prob_col) or row.get(f"Pred{prob_col.split('_')[0]}")
            o = row.get(odds_col) or row.get(f"Odds{prob_col.split('_')[0]}")
            try:
                probs.append(float(p))
                odds_vals.append(float(o))
            except (TypeError, ValueError):
                break
        if len(probs) != 3 or any(o <= 1 for o in odds_vals):
            continue
        # Normalize probabilities
        total = sum(probs)
        probs = [p / total for p in probs]
        # Check arbitrage
        arb = find_arbitrage([odds_vals], outcome_labels=["home", "draw", "away"])
        if arb.has_arbitrage:
            rows.append({
                "game": f"{away} @ {home}", "type": "arbitrage",
                "return": arb.guaranteed_return,
            })
            continue
        # Check value
        value_bets = find_value_bets(probs, odds_vals)
        for vb in value_bets:
            rows.append({
                "game": f"{away} @ {home}", "type": "value",
                "outcome": vb.outcome, "edge": round(vb.edge, 4),
                "ev": round(vb.expected_value, 4), "tier": vb.tier,
                "kelly_stake": round(vb.kelly_stake, 4),
            })

    result = pd.DataFrame(rows)
    output = data_dir / "value"
    output.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output / "daily_scan.parquet", index=False)
    print(f"Found {len(result)} opportunities")
    print(f"Saved to {output / 'daily_scan.parquet'}")


if __name__ == "__main__":
    main()
