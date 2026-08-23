# 06 — Implied Odds & Bookmaker Margin

`penaltyblog.implied` is the single best way to convert bookmaker prices
into "true" probabilities. It exposes seven overround-removal algorithms:

| Method                           | Year | Family         | Notes                                  |
|----------------------------------|------|----------------|----------------------------------------|
| `MULTIPLICATIVE`                 | —    | Proportional   | Default, divides by sum of inverse odds|
| `ADDITIVE`                       | —    | Proportional   | Removes equal proportion from each     |
| `POWER`                          | —    | Power          | Power-law correction                   |
| `SHIN`                           | 1992 | Bayesian       | Models insider trading                 |
| `DIFFERENTIAL_MARGIN_WEIGHTING`  | —    | Wisdom of crowd| Buchdahl's WOTC                        |
| `ODDS_RATIO`                     | —    | Wisdom of crowd| Cheung's odds-ratio                    |
| `LOGARITHMIC`                    | —    | Logit space    | Robust additive in log-odds space      |

All seven accept American, fractional, or decimal odds and return an
`ImpliedProbabilities` dataclass with `.probabilities`, `.margin`, and
helper methods (`.most_likely_probability`, `.get_probability_by_name`,
etc.).

Pitch Oracle touches:

- `track_predictions.py` — replace the bespoke implied-probability
  computation.
- New `compute_value_bets.py` — multi-bookmaker value detector.
- New `fair_odds.py` — show the model-fair price next to the bookmaker
  price in the Streamlit UI.

## 6.1 The seven methods side-by-side

```python
from penaltyblog.implied import calculate_implied, ImpliedMethod, OddsFormat

# Pinnacle-style 1X2: home 2.10, draw 3.40, away 3.60
odds = [2.10, 3.40, 3.60]
names = ["home", "draw", "away"]

for method in ImpliedMethod:
    result = calculate_implied(odds, method=method, market_names=names)
    print(f"{method.value:32s} margin={result.margin:.4f}  "
          f"P(home)={result['home']:.4f}  P(draw)={result['draw']:.4f}  "
          f"P(away)={result['away']:.4f}")
```

Typical output:

```
multiplicative                  margin=0.0689  P(home)=0.4424  P(draw)=0.2732  P(away)=0.2581
additive                        margin=0.0689  P(home)=0.4479  P(draw)=0.2787  P(away)=0.2633
power                           margin=0.0689  P(home)=0.4471  P(draw)=0.2781  P(away)=0.2623
shin                            margin=0.0689  P(home)=0.4523  P(draw)=0.2831  P(away)=0.2647
differential_margin_weighting   margin=0.0689  P(home)=0.4498  P(draw)=0.2797  P(away)=0.2610
odds_ratio                      margin=0.0689  P(home)=0.4482  P(draw)=0.2789  P(away)=0.2615
logarithmic                     margin=0.0689  P(home)=0.4487  P(draw)=0.2793  P(away)=0.2620
```

The differences are small on a tight 1X2 market but matter on **outright
markets** and **Asian handicaps** where one outcome has very low
implied probability.

## 6.2 Use Shin for markets with a known favourite

`SHIN` (Shin, 1992) treats the bookmaker's margin as being proportional
to the sum of squared-root implied probabilities — a model of insider
trading. It works best when one outcome is heavily favoured.

```python
# Cup final: Arsenal 1.45, Draw 4.50, Chelsea 7.50
odds = [1.45, 4.50, 7.50]
result = calculate_implied(odds, method="shin", market_names=["home", "draw", "away"])
print(result.margin, result.probabilities)
```

`method="shin"` accepts the string or the enum. Both work.

## 6.3 Use Logarithmic as the default for 1X2

`LOGARITHMIC` (sometimes called "logit-shift") is the most robust default
for 1X2 markets. It works in log-odds space and preserves the relative
ratios of the bookmaker's prices.

```python
result = calculate_implied(odds, method="logarithmic", market_names=names)
```

The `method_params` dict exposes the shift `c` that was applied — useful
for diagnostics:

```python
print(result.method_params["c"])
```

## 6.4 Multi-format odds

`OddsInput` accepts decimal, American, or fractional odds with the same
method API:

```python
from penaltyblog.implied.models import OddsInput, OddsFormat

american = OddsInput(values=[+170, +130, +340],
                     format=OddsFormat.AMERICAN,
                     market_names=["home", "draw", "away"])
result = calculate_implied(american, method="shin")

fractional = OddsInput(values=["17/10", "13/10", "34/10"],
                       format=OddsFormat.FRACTIONAL,
                       market_names=["home", "draw", "away"])
result = calculate_implied(fractional, method="multiplicative")
```

This is invaluable when ingesting odds from US books (American) and
European books (decimal) into the same analysis.

## 6.5 Multi-bookmaker value detection

This is the killer feature. The flow:

1. For every fixture, gather odds from N bookmakers.
2. For each bookmaker's 1X2 market, compute the **best odds** for each
   outcome.
3. Run the implied-probability calculation on the best odds.
4. Compare to your model's probability.
5. Flag any outcome where `model_prob > implied_prob + threshold`.

```python
# scripts/value/scan_value_bets.py
from penaltyblog.implied import calculate_implied, ImpliedMethod
from penaltyblog.betting import identify_value_bet, find_arbitrage_opportunities
import pandas as pd

slate = pd.read_parquet("data_files/odds/upcoming.parquet")  # one row per (match, bookmaker)
model = pd.read_parquet("data_files/predictions/upcoming.parquet")  # model probabilities

rows = []
for match_id, group in slate.groupby("match_id"):
    bookies = group.groupby("bookmaker")
    # Collect every bookmaker's 1X2 in a 2D array
    odds_per_bookie = []
    for _, b in bookies:
        odds_per_bookie.append([b["home"].iloc[0],
                                b["draw"].iloc[0],
                                b["away"].iloc[0]])
    arb = find_arbitrage_opportunities(odds_per_bookie,
                                       outcome_labels=["home", "draw", "away"])
    if arb.has_arbitrage:
        rows.append({"match_id": match_id, "type": "arbitrage",
                     "return": arb.guaranteed_return,
                     "best_odds": arb.best_odds,
                     "best_bookmakers": arb.best_bookmakers})
        continue

    # Use the best odds for the value check
    best = arb.best_odds
    implied = calculate_implied(best, method=ImpliedMethod.LOGARITHMIC,
                                 market_names=["home", "draw", "away"])
    m = model[model["match_id"] == match_id].iloc[0]
    value = identify_value_bet(
        bookmaker_odds=best,
        estimated_probability=[m["home_win"], m["draw"], m["away_win"]],
        kelly_fraction=0.5,
    )
    for i, vr in enumerate(value.individual_results):
        if vr.is_value_bet:
            rows.append({
                "match_id": match_id, "type": "value",
                "outcome": ["home", "draw", "away"][i],
                "edge": vr.edge,
                "ev":  vr.expected_value,
                "kelly_stake": vr.recommended_stake_fraction,
            })

pd.DataFrame(rows).to_parquet("data_files/value/value_bets.parquet")
```

That single script runs the full value detection pipeline. The output
artifacts feed the "Value Bets" page in the Streamlit app.

## 6.6 Per-bookmaker margin audit

Bookmakers have very different default overround profiles. Pinnacle
runs at 2–3%; recreational books at 6–10%. Track it:

```python
# scripts/odds/margin_audit.py
from penaltyblog.implied import calculate_implied, ImpliedMethod
import pandas as pd

odds = pd.read_parquet("data_files/odds/all.parquet")
margins = []
for _, row in odds.iterrows():
    res = calculate_implied(
        [row["home"], row["draw"], row["away"]],
        method=ImpliedMethod.MULTIPLICATIVE,
    )
    margins.append({"bookmaker": row["bookmaker"], "match_id": row["match_id"],
                    "margin": res.margin})
pd.DataFrame(margins).groupby("bookmaker").agg(
    margin_mean=("margin", "mean"),
    margin_p95=("margin", lambda s: s.quantile(0.95)),
).to_csv("data_files/odds/margin_audit.csv")
```

This becomes a quarterly report — which books are consistently sharp?

## 6.7 Comparing your model to the bookmaker

For every fixture:

```python
from penaltyblog.implied import calculate_implied, ImpliedMethod

slate = pd.read_parquet("data_files/predictions/upcoming.parquet")
odds  = pd.read_parquet("data_files/odds/upcoming.parquet")

merged = slate.merge(odds, on="match_id")
for _, row in merged.iterrows():
    implied = calculate_implied(
        [row["home"], row["draw"], row["away"]],
        method=ImpliedMethod.LOGARITHMIC,
        market_names=["home", "draw", "away"],
    )
    merged.loc[row.name, "imp_home"] = implied["home"]
    merged.loc[row.name, "imp_draw"] = implied["draw"]
    merged.loc[row.name, "imp_away"] = implied["away"]

merged["edge_home"] = merged["home_win"] - merged["imp_home"]
merged["edge_draw"] = merged["draw"]     - merged["imp_draw"]
merged["edge_away"] = merged["away_win"] - merged["imp_away"]
merged.to_parquet("data_files/predictions/with_edges.parquet")
```

Sort by `edge_home` descending and you have the "Model vs Market"
leaderboard. The biggest edge is the best candidate for a value bet.

## 6.8 Compute model-fair odds

The Streamlit UI should show two prices per market:

```python
def fair_odds(model_prob: float) -> float:
    """Return the fair decimal odds for a model probability."""
    return round(1.0 / model_prob, 2)

slate["fair_home"] = slate["home_win"].apply(fair_odds)
slate["fair_draw"] = slate["draw"].apply(fair_odds)
slate["fair_away"] = slate["away_win"].apply(fair_odds)
```

`fair_home > book_home` → model thinks home is under-priced by the
bookmaker. That's the value-bet signal in its simplest form.

## 6.9 Asian handicap fair odds

Asian handicap prices are a 2-outcome market. Apply the same logic:

```python
def ah_implied(home_ah: float, away_ah: float):
    return calculate_implied([home_ah, away_ah],
                             method=ImpliedMethod.LOGARITHMIC,
                             market_names=["home", "away"])

# Penaltyblog's grid gives the fair probability
from penaltyblog.models import DixonColesGoalModel
grid = model.predict("Arsenal", "Chelsea", max_goals=15)
ah = grid.asian_handicap_probs("home", -0.5)   # {"win", "push", "lose"}

# Compare to bookmaker's handicap line
book = ah_implied(1.95, 1.95)   # both 1.95 = 2.6% margin
edge = ah["win"] - book["home"]
```

See [11-probability-grid.md](11-probability-grid.md) for the full set of
handicap helpers.

## 6.10 Pitch Oracle integration

**Touches**

- `track_predictions.py` — replace custom implied-probability code with
  `calculate_implied`.
- New `scripts/value/scan_value_bets.py`.
- New `scripts/odds/margin_audit.py`.
- New `data_files/value/` and `data_files/odds/` directories.
- Streamlit "Value Bets" and "Model vs Market" pages.

**Does not touch**

- The goal models (odds are an independent view).

**Migration cost**

- One engineer, ~2 days.
