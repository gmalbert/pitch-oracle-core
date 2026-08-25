# 11 — FootballProbabilityGrid

`FootballProbabilityGrid` is the unsung hero of the library. It wraps a
2D `goal_matrix` (where `[h, a]` is `P(Home = h, Away = a)`) and exposes
30+ derived markets for free:

- 1X2 (home win, draw, away win)
- Double chance (1X, X2, 12)
- Both teams to score (BTTS yes / no)
- Totals (over / under / push) for any line including quarter lines
- Asian handicap (win / push / lose) for any line including quarter
  lines
- Draw No Bet (home / away)
- Win to nil (home / away)
- Exact score lookup
- Marginal distributions (home goals, away goals, total goals)
- Expected points (3/1/0)

Every goal model in [03-goal-models.md](03-goal-models.md) returns a
`FootballProbabilityGrid` from `.predict()`. Even the
`create_dixon_coles_grid()` helper accepts external lambdas (so you can
plug in XGBoost-predicted expected goals instead of using a fitted
Dixon-Coles model).

## 11.1 Core API

```python
grid = model.predict("Arsenal", "Chelsea", max_goals=15)

# 1X2
g.home_win       # P(H > A)
g.draw           # P(H = A)
g.away_win       # P(H < A)
g.home_draw_away # [P(H), P(D), P(A)]

# BTTS
g.btts_yes       # P(H > 0 and A > 0)
g.btts_no        # 1 - btts_yes

# Double chance
g.double_chance_1x   # P(H or D)
g.double_chance_x2   # P(D or A)
g.double_chance_12   # P(H or A)

# Draw no bet
g.draw_no_bet_home   # P(H wins | not draw)
g.draw_no_bet_away   # P(A wins | not draw)

# Win to nil
g.win_to_nil_home    # P(H > 0, A == 0)
g.win_to_nil_away    # P(H == 0, A > 0)

# Expected points
g.expected_points_home()  # 3 * P(H) + 1 * P(D)
g.expected_points_away()  # 3 * P(A) + 1 * P(D)
```

## 11.2 Totals

```python
under, push, over = g.totals(2.5)
# (0.32, 0.0, 0.68)

under, push, over = g.totals(2.0)
# (0.18, 0.21, 0.61)   # integer line: push is meaningful

under, push, over = g.totals(2.25)
# (0.27, 0.10, 0.63)   # quarter line: split stake
```

The push probability is non-zero only for integer lines. Quarter lines
(2.25, 2.75, …) automatically split the stake between the adjacent half
lines.

### Convenience: under/over without push

```python
g.total_goals("under", 2.5)   # P(total < 2.5)
g.total_goals("over",  2.5)   # P(total > 2.5)
```

Use this when you only care about the win/lose sides, not the push.

## 11.3 Asian handicap

```python
probs = g.asian_handicap_probs("home", -0.5)
# {"win": 0.471, "push": 0.0, "lose": 0.529}

probs = g.asian_handicap_probs("home", -0.25)
# {"win": ..., "push": ..., "lose": ...}
# Quarter line: stake split between 0.0 and -0.5

probs = g.asian_handicap_probs("home", -1.0)
# {"win": ..., "push": ..., "lose": ...}
# Integer line: push meaningful
```

Convenience wrapper:

```python
g.asian_handicap("home", -0.5)    # just the win probability
g.asian_handicap("away",  0.25)   # away win probability at +0.25
```

## 11.4 Score distributions

```python
# Exact score probability
g.exact_score(1, 1)        # P(1-1)
g.exact_score(0, 0)        # P(0-0)

# Marginal distributions (over all goals 0..max)
g.home_goal_distribution()    # array: P(H = 0), P(H = 1), ...
g.away_goal_distribution()
g.total_goals_distribution()

# Most likely scoreline
import numpy as np
home_pmf = g.home_goal_distribution()
away_pmf = g.away_goal_distribution()
score_grid = np.outer(home_pmf, away_pmf)
h, a = np.unravel_index(score_grid.argmax(), score_grid.shape)
print(f"Most likely: {h}-{a} ({score_grid[h, a]:.3f})")
```

## 11.5 Build a grid from external lambdas

Sometimes you don't want a fitted goal model — you have your own expected
goals from an XGBoost or neural net. `create_dixon_coles_grid()` accepts
the lambdas directly:

```python
from penaltyblog.models.football_probability_grid import create_dixon_coles_grid

# XGBoost output
xg = model.predict_xg(...)   # (home_xg, away_xg)

grid = create_dixon_coles_grid(
    home_lambda=xg["home_xg"],
    away_lambda=xg["away_xg"],
    rho=-0.05,                # the fitted league-average
    max_goals=15,
)

# Now you have every market above
print(grid.home_win, grid.btts_yes, grid.asian_handicap("home", -0.5))
```

This is the bridge between Pitch Oracle's existing ML ensemble and the
goal-model market layer. Use it to:

1. Train an XGBoost on event-level features.
2. Predict `(home_xg, away_xg)` per match.
3. Pass to `create_dixon_coles_grid()` to get the full market surface.
4. Compare with the MLE model's market surface as an ensemble member.

## 11.6 Vectorised grids across a slate

The point of `model.predict_many()` is that it allocates one shared 3D
grid and avoids the Python-level overhead:

```python
grids = model.predict_many(home_idx, away_idx, max_goals=15)

slate["home_win"]  = [g.home_win for g in grids]
slate["draw"]      = [g.draw     for g in grids]
slate["away_win"]  = [g.away_win for g in grids]
slate["btts"]      = [g.btts_yes for g in grids]
slate["over_2_5"]  = [g.total_goals("over", 2.5) for g in grids]
slate["ah_minus_half_home"] = [g.asian_handicap("home", -0.5) for g in grids]
```

For a 50-fixture slate, this is a single Cython call returning 50 grids.

## 11.7 The "every market at once" pattern

A common request: "show me every market for this fixture in one
DataFrame". Here it is:

```python
def market_grid(model, home, away, max_goals=15) -> pd.DataFrame:
    g = model.predict(home, away, max_goals=max_goals)
    rows = [
        ("1X2",   "Home",  g.home_win),
        ("1X2",   "Draw",  g.draw),
        ("1X2",   "Away",  g.away_win),
        ("DC",    "1X",    g.double_chance_1x),
        ("DC",    "X2",    g.double_chance_x2),
        ("DC",    "12",    g.double_chance_12),
        ("BTTS",  "Yes",   g.btts_yes),
        ("BTTS",  "No",    g.btts_no),
        ("DNB",   "Home",  g.draw_no_bet_home),
        ("DNB",   "Away",  g.draw_no_bet_away),
        ("WTN",   "Home",  g.win_to_nil_home),
        ("WTN",   "Away",  g.win_to_nil_away),
    ]
    for line in [1.5, 2.5, 3.5, 4.5]:
        u, _, o = g.totals(line)
        rows += [(f"Totals {line}", "Under", u),
                 (f"Totals {line}", "Over",  o)]
    for line in [-1.5, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.5]:
        ah = g.asian_handicap_probs("home", line)
        rows += [(f"AH {line:+}", "Win",  ah["win"]),
                 (f"AH {line:+}", "Push", ah["push"]),
                 (f"AH {line:+}", "Lose", ah["lose"])]
    return pd.DataFrame(rows, columns=["market", "side", "probability"])
```

That's the canonical "fixture detail" view for the Streamlit page.

## 11.8 Use grid properties for calibration

```python
# For each match, log the model's H win probability and the actual result
records = []
for _, row in test_df.iterrows():
    g = model.predict(row["team_home"], row["team_away"])
    records.append({"predicted_p_home": g.home_win,
                    "actual_home_win": row["result_H"]})
calibration = pd.DataFrame(records)

# Bin into deciles
calibration["decile"] = pd.qcut(calibration["predicted_p_home"], 10)
print(calibration.groupby("decile")["actual_home_win"].agg(["mean", "count"]))
```

That's the reliability diagram. If the actual mean within each decile
matches the predicted midpoint, the model is calibrated.

## 11.9 Compute market consensus

Bookmaker markets are 3-outcome (1X2), 2-outcome (BTTS, DNB, WTN), or
continuous (Asian handicap, totals). For 3-outcome, the "true" probability
is the implied (de-vigged) probability. For 2-outcome, build a synthetic
two-outcome "market" and apply the same logic:

```python
def market_consensus(g: FootballProbabilityGrid,
                     book_home: float, book_draw: float, book_away: float) -> dict:
    """Return the model-vs-book disagreement for every 1X2 outcome."""
    from penaltyblog.implied import calculate_implied, ImpliedMethod
    implied = calculate_implied(
        [book_home, book_draw, book_away],
        method=ImpliedMethod.LOGARITHMIC,
        market_names=["home", "draw", "away"],
    )
    return {
        "home": g.home_win - implied["home"],
        "draw": g.draw     - implied["draw"],
        "away": g.away_win - implied["away"],
    }
```

This is the unit of analysis for the "Value Bets" page.

## 11.10 Fastest path: a minimal fixture detail page

```python
# streamlit_pages/03_fixture_detail.py
import streamlit as st
from penaltyblog.models import DixonColesGoalModel

@st.cache_resource
def load_model():
    return DixonColesGoalModel.load("models/dixon_coles_2425.pkl")

model = load_model()
home = st.selectbox("Home", model.teams)
away = st.selectbox("Away", [t for t in model.teams if t != home])

g = model.predict(home, away, max_goals=15)
st.metric("Home win", f"{g.home_win:.1%}")
st.metric("Draw",     f"{g.draw:.1%}")
st.metric("Away win", f"{g.away_win:.1%}")

st.subheader("Goals")
st.write(f"Expected: {g.home_goal_expectation:.2f} - {g.away_goal_expectation:.2f}")

st.subheader("Markets")
st.dataframe(market_grid(model, home, away), use_container_width=True)
```

That's a complete page in 30 lines. The model is loaded once via
`@st.cache_resource`; the markets are computed on demand.

## 11.11 Pitch Oracle integration

**Touches**

- New `pitch_oracle_core/markets.py` — wraps `FootballProbabilityGrid`
  with helpers like `market_grid()`.
- Streamlit pages that currently compute 1X2 / totals / BTTS by hand:
  `streamlit_pages/02_match_predictions.py`,
  `streamlit_pages/03_fixture_detail.py`, `04_match_centre.py`.
- New `scripts/predictions/build_grid_features.py` — for every
  upcoming fixture, dump the full grid into a parquet.

**Does not touch**

- The modelling layer (grids are returned automatically).

**Migration cost**

- One engineer, ~2 days including the markets helper module.
