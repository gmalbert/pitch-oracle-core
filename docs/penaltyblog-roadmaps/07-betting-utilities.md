# 07 — Betting Utilities

`penaltyblog.betting` is a small but high-quality module that turns a
model probability and a bookmaker price into an actionable bet:

| Function                      | Use                                          |
|-------------------------------|----------------------------------------------|
| `kelly_criterion`             | Single bet sizing                            |
| `multiple_kelly_criterion`    | 3-way market sizing, simultaneous or independent |
| `identify_value_bet`          | Single or batch value detection              |
| `find_arbitrage_opportunities`| Multi-bookmaker arbitrage scanner            |
| `arbitrage_hedge`             | Existing-position hedge stakes               |
| `convert_odds`                | Decimal ↔ fractional ↔ American              |

Pitch Oracle currently has no betting-strategy module; this roadmap
describes the missing piece.

## 7.1 Single-bet Kelly

```python
from penaltyblog.betting import kelly_criterion

res = kelly_criterion(
    decimal_odds=2.10,         # Pinnacle
    true_prob=0.55,            # your model
    fraction=0.5,              # half-Kelly, conservative
)
print(f"Stake:        {res.stake:.2%}")
print(f"Expected growth: {res.expected_growth:.4%}")
print(f"Edge:         {res.edge:+.3f}")
print(f"Favourable:   {res.is_favorable}")
print(f"Risk of ruin: {res.risk_of_ruin:.4f}")
print(res.risk_metrics)
# RiskMetrics(expected_profit=..., expected_return=...,
#             kelly_growth_rate=..., wealth_volatility=...,
#             sharpe_ratio=..., probability_of_ruin=...,
#             value_at_risk_95=..., ...)
```

`kelly_criterion` returns a `KellyResult` dataclass with every diagnostic
a bettor needs:

- `stake` — fraction of bankroll.
- `expected_growth` — log-growth rate (what Kelly actually optimises).
- `edge` — `(true_prob * odds) - 1`, the per-unit expected value.
- `risk_metrics` — full distribution-aware metrics for the resulting
  bankroll distribution.

### Scalar vs vector

`kelly_criterion` is fully vectorised:

```python
import numpy as np
odds = np.array([2.10, 3.40, 3.60])
probs = np.array([0.55, 0.27, 0.18])
res = kelly_criterion(odds, probs, fraction=0.5)
print(res.stake)        # array of stakes
print(res.is_favorable) # array of bools
```

## 7.2 3-way market Kelly (1X2)

For a 1X2 market, the three outcomes are **mutually exclusive** — only
one pays out. `multiple_kelly_criterion` solves the joint Kelly problem:

```python
from penaltyblog.betting import multiple_kelly_criterion

res = multiple_kelly_criterion(
    decimal_odds=[2.10, 3.40, 3.60],   # home, draw, away
    true_probs=[0.55, 0.27, 0.18],
    fraction=0.5,
    max_total_stake=0.10,
    method="simultaneous",   # or "independent"
)
print(f"Stakes: {res.stakes}")
print(f"Total stake: {res.total_stake:.2%}")
print(f"Portfolio edge: {res.portfolio_edge:.3f}")
print(f"Sharpe: {res.risk_metrics.sharpe_ratio:.3f}")
```

`method="simultaneous"` solves a constrained optimisation that
maximises the joint expected log-growth. `method="independent"` falls
back to per-outcome Kelly scaled to `max_total_stake`; it's faster but
sub-optimal when outcomes are mutually exclusive.

## 7.3 Value-bet identification

```python
from penaltyblog.betting import identify_value_bet

# Single
res = identify_value_bet(
    bookmaker_odds=2.10,
    estimated_probability=0.55,
    kelly_fraction=0.5,
)
print(res.expected_value, res.is_value_bet, res.recommended_stake_fraction)

# Multiple
res = identify_value_bet(
    bookmaker_odds=[2.10, 3.40, 3.60],
    estimated_probability=[0.55, 0.27, 0.18],
    kelly_fraction=0.5,
)
print(res.total_value_bets, res.best_edge, res.worst_edge)
```

`MultipleValueBetResult` exposes:

- `individual_results` — list of per-bet `ValueBetResult`s.
- `total_value_bets`, `average_edge`, `total_expected_value`.
- `kelly_stakes`, `total_kelly_stake`, `portfolio_expected_return`.
- `best_value_bet_index`, `best_edge`, `worst_edge`.

## 7.4 Arbitrage detection

```python
from penaltyblog.betting import find_arbitrage_opportunities

# Two bookmakers, two outcomes
arb = find_arbitrage_opportunities(
    bookmaker_odds_list=[
        [2.10, 1.85],   # bookie A
        [1.95, 2.00],   # bookie B
    ],
    outcome_labels=["home", "away"],
)
print(arb.has_arbitrage, arb.guaranteed_return, arb.stake_percentages)
```

If `arb.has_arbitrage` is `True`, `stake_percentages` tells you exactly
how to split a bankroll across the best odds. `arbitrage_margin` is the
guaranteed return (e.g. 0.018 = 1.8%).

### Three-way arbitrage

```python
# Three bookmakers, three outcomes
arb = find_arbitrage_opportunities(
    bookmaker_odds_list=[
        [2.10, 3.40, 3.60],
        [2.05, 3.50, 3.55],
        [2.12, 3.45, 3.40],
    ],
    outcome_labels=["home", "draw", "away"],
)
print(arb.has_arbitrage, arb.best_odds, arb.best_bookmakers)
```

`best_bookmakers` is a list of indices into the input — useful for
routing the bet to the right book.

## 7.5 Hedging an existing position

Suppose you backed Arsenal at 3.50 for £100 and they made the final.
You want to lock in profit by laying Chelsea:

```python
from penaltyblog.betting import arbitrage_hedge

res = arbitrage_hedge(
    existing_stakes=[100, 0, 0],          # £100 on Arsenal
    existing_odds=[3.50, 4.20, 2.80],     # original prices
    hedge_odds=[3.40, 4.10, 2.75],        # current prices
)
print(res.practical_hedge_stakes)  # how much to lay on each outcome
print(res.guaranteed_profit)        # guaranteed profit (or loss)
```

For asymmetric cases, `guaranteed_profit` is the **worst-case** profit
across all outcomes. Use `hedge_all=False` to only hedge outcomes with
existing exposure, and `allow_lay=True` to permit negative (lay) stakes
in the raw output.

### Real-time in-play hedge

```python
# In-play: Arsenal leading 1-0, 70th minute
hedge_odds = [4.50, 1.50, 2.10]   # draw collapsed, Chelsea shortened
res = arbitrage_hedge(
    existing_stakes=[100, 0, 0],
    existing_odds=[3.50, 4.20, 2.80],
    hedge_odds=hedge_odds,
    target_profit=20.0,             # lock in £20 profit
)
print(res.practical_hedge_stakes)
```

## 7.6 Odds conversion

```python
from penaltyblog.betting import convert_odds

# Decimal → American
print(convert_odds(2.50, from_format="decimal", to_format="american"))
# +150

# American → Decimal
print(convert_odds(-150, from_format="american", to_format="decimal"))
# 1.667

# Fractional → Decimal
print(convert_odds("3/1", from_format="fractional", to_format="decimal"))
# 4.0
```

The function accepts both numeric and string inputs, so the same call
works whether your upstream feed gives American integers or fractional
strings.

## 7.7 The full Pitch Oracle value scanner

A single script that ties it all together:

```python
# scripts/value/daily_value_scan.py
from penaltyblog.implied import calculate_implied, ImpliedMethod
from penaltyblog.betting import (
    identify_value_bet, find_arbitrage_opportunities,
    multiple_kelly_criterion, kelly_criterion,
)
import pandas as pd

slate  = pd.read_parquet("data_files/predictions/upcoming.parquet")
odds   = pd.read_parquet("data_files/odds/upcoming.parquet")
merged = slate.merge(odds, on="match_id", suffixes=("_model", "_book"))

bankroll = 1000.0
records = []

for match_id, group in merged.groupby("match_id"):
    bookies = group.groupby("bookmaker")
    odds_per_bookie = [list(b[["home", "draw", "away"]].iloc[0])
                       for _, b in bookies]

    arb = find_arbitrage_opportunities(
        odds_per_bookie, outcome_labels=["home", "draw", "away"]
    )
    if arb.has_arbitrage:
        stake = bankroll * 0.02      # cap arb exposure at 2% bankroll
        records.append({
            "match_id": match_id,
            "type": "arbitrage",
            "return_pct": arb.guaranteed_return,
            "stake": stake,
            "expected_profit": stake * arb.guaranteed_return,
            "best_odds": arb.best_odds,
            "best_bookmakers": arb.best_bookmakers,
        })
        continue

    m = slate[slate["match_id"] == match_id].iloc[0]
    implied = calculate_implied(arb.best_odds, method=ImpliedMethod.LOGARITHMIC,
                                 market_names=["home", "draw", "away"])
    kelly = kelly_criterion(
        arb.best_odds,
        [m["home_win"], m["draw"], m["away_win"]],
        fraction=0.25,
    )
    if not kelly.is_favorable:
        continue
    records.append({
        "match_id": match_id,
        "type": "value",
        "best_odds": arb.best_odds,
        "best_bookmakers": arb.best_bookmakers,
        "stake_pct": float(kelly.stake),
        "stake_gbp": bankroll * float(kelly.stake),
        "expected_growth": float(kelly.expected_growth),
        "edge": float(kelly.edge),
    })

pd.DataFrame(records).to_parquet("data_files/value/daily_scan.parquet")
```

Run this daily. It produces a single parquet file that the Streamlit
"Today's Bets" page reads.

## 7.8 Streamlit "Today's Bets" page

```python
# streamlit_pages/06_today_bets.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Today's Bets", layout="wide")
st.title("Today's Bets")

bets = pd.read_parquet("data_files/value/daily_scan.parquet")
if bets.empty:
    st.info("No value or arbitrage opportunities today.")
else:
    arb = bets[bets["type"] == "arbitrage"]
    val = bets[bets["type"] == "value"]
    st.subheader("Arbitrage")
    st.dataframe(arb, use_container_width=True)
    st.subheader("Value")
    st.dataframe(val, use_container_width=True)
```

## 7.9 Responsible staking

Always present Kelly with a fractional override:

```python
KELLY_FRACTION = 0.25    # quarter-Kelly, even more conservative than half

def kelly_safe(odds, prob, fraction=KELLY_FRACTION):
    return kelly_criterion(odds, prob, fraction=fraction)
```

Pin this in a config file; the Streamlit page exposes it as a slider so
the user can dial the aggressiveness up or down. Document the math in a
sidebar so users understand why quarter-Kelly is the default.

## 7.10 Common pitfalls

1. **Don't use Kelly on overround-positive prices without de-vigging.**
   Always feed `model_prob`, not `implied_prob`.
2. **Don't use Kelly with `fraction=1.0` in production.** Even 0.5 is
   aggressive; the literature suggests 0.25–0.5 is the safe range.
3. **Don't treat arbitrage as risk-free.** Bookmaker limits, account
   restrictions, and withdrawal friction eat the margin in practice.
4. **Don't chain Kelly + Kelly.** A "value bet" already includes the
   Kelly stake; do not run Kelly twice.

## 7.11 Pitch Oracle integration

**Touches**

- New `scripts/value/daily_value_scan.py` — the orchestrator.
- New `scripts/odds/margin_audit.py` — bookmaker-margin dashboard.
- New `streamlit_pages/06_today_bets.py` — the consumer.
- New `data_files/value/` and `data_files/odds/`.

**Does not touch**

- The goal models. The betting layer reads model probabilities and
  bookmaker odds as inputs.

**Migration cost**

- One engineer, ~3 days including the responsible-staking copy and the
  Streamlit page.
