# 05 — Ratings Systems

`penaltyblog.ratings` contains four team-strength systems:

| Class            | Output                                | Best for                            |
|------------------|---------------------------------------|-------------------------------------|
| `Elo`            | Per-team scalar rating                | Live updating, simple league tables |
| `Massey`         | Rating + offence + defence            | CFB-style power rankings            |
| `Colley`         | Per-team scalar rating                | Schedule-strength adjusted tables   |
| `PiRatingSystem` | Per-team (home, away) pair            | High-information match prediction   |

Together they replace the bits of Pitch Oracle that:

- Pull `ClubElo` ratings and pass them through to models (now `ClubElo`
  + `Elo` + `PiRatingSystem`).
- Compute league tables with goal-difference tiebreakers (now `Massey`).
- Build power rankings for the "Power Rankings" page (now `Massey` /
  `Colley`).

## 5.1 Elo — the simplest live-updating rating

`Elo` is a Python port of the classic Elo algorithm, with two
football-specific tweaks: a configurable **home field advantage**
(measured in Elo points) and a Gaussian-shaped **draw probability**
centred at zero Elo difference.

```python
from penaltyblog.ratings import Elo
import pandas as pd

df = pd.read_parquet("data_files/features/training_set.parquet").sort_values("datetime")

elo = Elo(k=20, home_field_advantage=100.0)
for _, row in df.iterrows():
    # 0 = home win, 1 = draw, 2 = away win
    if   row["goals_home"] > row["goals_away"]: result = 0
    elif row["goals_home"] < row["goals_away"]: result = 2
    else:                                       result = 1
    elo.update_ratings(row["team_home"], row["team_away"], result)

ratings = pd.Series(elo.ratings).sort_values(ascending=False)
print(ratings.head(10))
```

### Predict a single match

```python
probs = elo.calculate_match_probabilities("Arsenal", "Chelsea")
# {"home_win": 0.48, "draw": 0.26, "away_win": 0.26}
```

The draw probability is `draw_base * exp(-(elo_diff**2) / (2*draw_width**2))`,
normalised. With the default `draw_base=0.30`, even mismatched teams
get ~10–15% draw probability — the same draw-floor we see in bookmaker
markets.

### Elo over time

```python
history = []
elo = Elo()
for _, row in df.iterrows():
    snapshot = dict(elo.ratings)
    snapshot["date"] = row["datetime"]
    history.append(snapshot)
    if   row["goals_home"] > row["goals_away"]: result = 0
    elif row["goals_home"] < row["goals_away"]: result = 2
    else:                                       result = 1
    elo.update_ratings(row["team_home"], row["team_away"], result)

elo_history = pd.DataFrame(history).set_index("date")
elo_history.to_parquet("data_files/ratings/elo_history.parquet")
```

That table feeds the "form" chart on each team page.

## 5.2 Massey — power rankings with offence and defence

The Massey method is a least-squares linear system. Each team's rating is
decomposed into an offensive and a defensive component. The system
matrix encodes how often each team played each other, with a normalisation
constraint to remove the trivial null space.

```python
from penaltyblog.ratings import Massey

massey = Massey(
    goals_home=df["goals_home"],
    goals_away=df["goals_away"],
    teams_home=df["team_home"],
    teams_away=df["team_away"],
)
rankings = massey.get_ratings()
print(ratings.sort_values("rating", ascending=False).head(10))
# Columns: team, rating, offence, defence
```

Massey ratings are not as live-update-friendly as Elo (they're
re-computed each season), but they have a clean interpretation:

- **`rating`** — combined strength.
- **`offence`** — goals scored above league average per game.
- **`defence`** — goals conceded below league average per game (positive
  means a good defence).

Use them for the "Power Rankings" page:

```python
rankings["rank"] = rankings["rating"].rank(ascending=False)
rankings.to_csv("data_files/ratings/massey_2425.csv", index=False)
```

## 5.3 Colley — schedule-strength adjusted rankings

Colley's method is also a linear system, but it uses a win/loss/draw
matrix weighted by `draw_weight`. The output is a scalar rating per
team, scaled so that 0.5 is "average" and 1.0 is "perfect".

```python
from penaltyblog.ratings import Colley

colley = Colley(
    goals_home=df["goals_home"],
    goals_away=df["goals_away"],
    teams_home=df["team_home"],
    teams_away=df["team_away"],
    include_draws=True,
    draw_weight=0.5,
)
ratings = colley.get_ratings().sort_values("rating", ascending=False)
print(ratings.head(10))
```

Colley is schedule-strength adjusted — a team that beats only other
strong teams ranks higher than a team that beat only weak teams, even
if they have the same record. This is exactly the right ranking for
mid-season "who deserves to be top" debates.

Use `draw_weight` to dial down the impact of draws:

```python
# A draw is worth a third of a win
colley = Colley(..., draw_weight=0.333)
```

## 5.4 Pi ratings — separate home and away strength

Pi ratings (Constantinou & Fenton, 2013) maintain a **separate home and
away rating** for every team. The rating update uses a "diminishing
error" function so a 6-0 win doesn't move the needle 6× more than a 1-0
win.

```python
from penaltyblog.ratings import PiRatingSystem

pi = PiRatingSystem(alpha=0.15, beta=0.10, k=0.75, sigma=1.0)
for _, row in df.iterrows():
    pi.update_ratings(
        home_team=row["team_home"],
        away_team=row["team_away"],
        observed_goal_difference=row["goals_home"] - row["goals_away"],
        date=row["datetime"],
    )

# Final per-team ratings
ratings = pd.DataFrame([
    {"team": team, "home": r["home"], "away": r["away"]}
    for team, r in pi.team_ratings.items()
])
ratings["avg"] = (ratings["home"] + ratings["away"]) / 2
print(ratings.sort_values("avg", ascending=False).head(10))
```

### Predict with Pi

```python
probs = pi.calculate_match_probabilities("Arsenal", "Chelsea")
# {"home_win": ..., "draw": ..., "away_win": ...}
```

The probability comes from a Normal distribution around the expected
goal difference, with a 0.5-goal draw margin.

### Inspect history

```python
history = pd.DataFrame(pi.rating_history)
# Columns: date, team, home_rating, away_rating
```

This gives you a per-team time series of home/away strength, perfect
for the "team profile" chart.

## 5.5 Compare all four rating systems

A single function that runs all four and stores the rankings side-by-side:

```python
# scripts/ratings/build_rankings.py
from penaltyblog.ratings import Elo, Massey, Colley, PiRatingSystem
import pandas as pd

df = pd.read_parquet("data_files/features/training_set.parquet").sort_values("datetime")

# --- Elo (live, single pass) ---
elo = Elo(k=20, home_field_advantage=100.0)
for _, row in df.iterrows():
    if   row["goals_home"] > row["goals_away"]: r = 0
    elif row["goals_home"] < row["goals_away"]: r = 2
    else:                                       r = 1
    elo.update_ratings(row["team_home"], row["team_away"], r)
elo_df = pd.DataFrame({"team": list(elo.ratings.keys()),
                       "rating": list(elo.ratings.values())})

# --- Massey / Colley (linear systems) ---
massey_df = Massey(df["goals_home"], df["goals_away"],
                   df["team_home"], df["team_away"]).get_ratings()
colley_df = Colley(df["goals_home"], df["goals_away"],
                   df["team_home"], df["team_away"]).get_ratings()

# --- Pi (separate home/away, single pass) ---
pi = PiRatingSystem()
for _, row in df.iterrows():
    pi.update_ratings(row["team_home"], row["team_away"],
                      row["goals_home"] - row["goals_away"],
                      date=row["datetime"])
pi_df = pd.DataFrame([
    {"team": t, "rating": (r["home"] + r["away"]) / 2}
    for t, r in pi.team_ratings.items()
])

# --- Merge + rank-normalise ---
out = (elo_df.rename(columns={"rating": "elo"})
              .merge(massey_df[["team", "rating"]].rename(columns={"rating": "massey"}),
                     on="team")
              .merge(colley_df.rename(columns={"rating": "colley"}), on="team")
              .merge(pi_df.rename(columns={"rating": "pi"}), on="team"))
for col in ("elo", "massey", "colley", "pi"):
    out[col + "_rank"] = out[col].rank(ascending=False)
out.to_parquet("data_files/ratings/all_rankings.parquet")
```

The result is a single table that can be sorted by any system, or by an
average rank. This is exactly the artifact the Streamlit "Power Rankings"
page should read.

## 5.6 Use ratings as model features

Every rating can become a feature for the ML ensemble:

```python
# scripts/ensemble/build_features.py
from penaltyblog.ratings import Elo, Massey, Colley, PiRatingSystem
import pandas as pd

# (fit ratings as above)
features = slate.merge(elo_df, left_on="team_home", right_on="team", how="left") \
                .rename(columns={"rating": "home_elo"}) \
                .drop(columns="team") \
                .merge(elo_df, left_on="team_away", right_on="team", how="left") \
                .rename(columns={"rating": "away_elo"}) \
                .drop(columns="team")
features["elo_diff"] = features["home_elo"] - features["away_elo"]
features["elo_diff_per_game"] = features["elo_diff"] / 400.0    # ≈ log-odds

# Same for Massey / Colley / Pi
features.to_parquet("data_files/features/ratings_features.parquet")
```

`elo_diff / 400.0` is the log-odds of a win under standard Elo; if the
ML model already knows that, the rating is redundant. If it doesn't
have that information elsewhere, the rating is a strong signal.

## 5.7 Pitch-side: live Elo for the "form" widget

Streamlit already has a per-team "form" widget that shows the last 5
results. Add an Elo delta over that window:

```python
from penaltyblog.ratings import Elo

elo = Elo()
# ... replay history ...
today = elo.ratings[team]
window_start = elo.ratings_at[team, today - 28 days]   # custom hook
delta = today - window_start
st.metric("Elo (28d)", f"{int(today)}", f"{int(delta):+d}")
```

`Elo.ratings` is a `Dict[str, float]`; if you also store `ratings_at`
during the replay loop, the UI can show the delta.

## 5.8 Ratings vs `ClubElo`

`ClubElo` is an **external** daily snapshot of Elo ratings maintained
by clubelo.com. `penaltyblog.ratings.Elo` is your **own** Elo, computed
on the fly from results.

Compare the two:

```python
from penaltyblog import ClubElo
from penaltyblog.ratings import Elo
import pandas as pd

external = ClubElo().get_elo_by_date()["elo"]
internal = pd.Series(elo.ratings)

joined = pd.concat([external.rename("clubelo"), internal.rename("po_elo")], axis=1)
joined["diff"] = joined["clubelo"] - joined["po_elo"]
joined.to_csv("data_files/ratings/elo_comparison.csv")

import matplotlib.pyplot as plt
joined.plot.scatter(x="clubelo", y="po_elo")
plt.plot([1000, 2000], [1000, 2000], "--", color="grey")
plt.title("Pitch Oracle Elo vs ClubElo.com")
plt.savefig("output/elo_comparison.png", dpi=150)
```

Persistent divergence between the two signals a calibration issue worth
investigating.

## 5.9 Pitch Oracle integration

**Touches**

- `analyze_team_form.py` — replace ad-hoc form scoring with Elo deltas.
- `fetch_clubelo.py` — keep; it's the *external* sanity check.
- New `scripts/ratings/build_rankings.py` to materialise the four-system
  leaderboard.
- New `data_files/ratings/` directory.
- New `scripts/ensemble/build_features.py` to add rating features.

**Does not touch**

- The goal models (ratings are an independent view).
- The Streamlit pages — they read the artifact.

**Migration cost**

- One engineer, ~2 days. The slowest part is wiring the artifact into
  every consumer repo's Streamlit page.
