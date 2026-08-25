# 08 — Fantasy Premier League

`penaltyblog.fpl` scrapes the official Fantasy Premier League API and
solves the lineup-optimisation problem with PuLP. It is the simplest
module to plug into the EPL consumer repo because it has zero
configuration beyond an internet connection.

The relevant Pitch Oracle touchpoints:

- The EPL consumer (`belgium-soccer`-style repo, currently scoped to
  `pitch-oracle-core`) needs a dedicated FPL workspace for users who
  also play FPL.
- `fetch_player_data.py` and `fetch_player_data_fd.py` can be partly
  replaced with `penaltyblog.fpl.get_player_data()`.
- A new `fpl/` workspace can host the optimiser, captain-pick advisor,
  and chip-strategy dashboard.

## 8.1 Gameweek helpers

```python
import penaltyblog as pb

# Current gameweek (the next unfinished one)
gw = pb.fpl.get_current_gameweek()
print(f"Current gameweek: {gw}")

# Full schedule of all 38 gameweeks (deadlines, finishes, is_current)
info = pb.fpl.get_gameweek_info()
print(info.columns.tolist())
# ['id', 'name', 'deadline_time', 'average_entry_score',
#  'finished', 'data_checked', 'highest_scoring_entry', ...
#   'is_current', 'is_next', 'is_previous', 'chip_plays', ...]
```

The `deadline_time` column is the most useful — it powers the "next
deadline" countdown in the Streamlit landing page.

## 8.2 Player master table

```python
players = pb.fpl.get_player_data()
print(players.columns.tolist())
```

The result is one row per FPL player with the columns FPL exposes:

```
id, first_name, second_name, web_name, team, team_name_short, position,
position_short, now_cost (in £m), total_points, form, points_per_game,
selected_by_percent, value_form, value_season, influence, creativity,
threat, ict_index, ...
```

`now_cost` is auto-scaled from tenths of £m to £m, the position and team
columns are merged in via the FPL element_types / teams endpoints.

This single DataFrame is the canonical player table for the entire FPL
workspace. Persist it once per refresh:

```python
players.to_parquet("data_files/fpl/players.parquet")
```

### Player ID mapping

```python
mapping = pb.fpl.get_player_id_mappings()
print(mapping.head())
#   first_name second_name   web_name  id
# 0   Mohamed     Salah     Mohamed Salah  277
```

Use `mapping` to wire UI lookups ("search by player name") to the FPL
player ID.

## 8.3 Per-player history

```python
history = pb.fpl.get_player_history(player_id=277)
print(history.columns.tolist())
# ['element', 'fixture', 'opponent_team', 'total_points', 'was_home',
#  'kickoff_time', 'team_h_score', 'team_a_score', 'minutes', 'goals_scored',
#  'assists', 'clean_sheets', 'goals_conceded', 'own_goals', 'penalties_saved',
#  'penalties_missed', 'yellow_cards', 'red_cards', 'saves', 'bonus',
#  'bps', 'influence', 'creativity', 'threat', 'ict_index', 'value', 'transfers_balance',
#  'selected', 'transfers_in', 'transfers_out']
```

This is the per-gameweek time series for one player. It feeds a per-player
form chart in the UI.

### Compute rolling form for all players

```python
import pandas as pd
import penaltyblog as pb

players = pb.fpl.get_player_data()
all_history = pd.concat([
    pb.fpl.get_player_history(pid).assign(player_id=pid)
    for pid in players["id"]
])
all_history.to_parquet("data_files/fpl/player_history.parquet")
```

For 700+ players this hits the FPL API ~700 times. Be polite — wrap in
a `ThreadPoolExecutor(max_workers=4)` with `requests` retry.

## 8.4 Optimal lineup via `optimise_team`

The headline feature. `optimise_team(formation, budget)` solves:

```
max   Σ  total_points[p] * x[p]
s.t.   Σ  cost[p] * x[p]          <= budget
       Σ  x[p] for GKPs           == 1
       Σ  x[p] for DEFs           == 5
       Σ  x[p] for MIDs           == 5
       Σ  x[p] for FWDs           == 3
       Σ  x[p] for any team       <= 3
       x[p] ∈ {0, 1}
```

```python
status, squad = pb.fpl.optimise_team(formation="2-5-5-3", budget=100)
print(status)             # 'Optimal'
print(f"Total points: {squad['total_points'].sum()}")
print(f"Total price:  £{squad['price'].sum():.1f}m")
print(squad)
```

The output is the chosen XI sorted by position then projected points.

### Try every formation, find the best

```python
formations = ["3-4-3", "3-5-2", "4-4-2", "4-3-3", "4-5-1",
              "5-3-2", "5-4-1", "2-5-5-3"]
results = []
for f in formations:
    try:
        status, squad = pb.fpl.optimise_team(formation=f, budget=100)
        if status == "Optimal":
            results.append({"formation": f, "points": squad["total_points"].sum(),
                            "price": squad["price"].sum()})
    except Exception:
        continue
best = max(results, key=lambda r: r["points"])
print(f"Best formation: {best['formation']} → {best['points']} pts")
```

This is a daily job — the result drives the "Should you change formation
this week?" advisor.

## 8.5 Custom optimiser with bonus features

`optimise_team` returns the basic XI. The real FPL game has captaincy
chips, vice-captain, free-hit, bench boost, wildcard, triple captain.
Extend the optimiser:

```python
# scripts/fpl/optimal_with_captain.py
import pulp
import pandas as pd
import penaltyblog as pb

players = pb.fpl.get_player_data()
formation = "3-4-3"
gk, df, md, fw = map(int, formation.split("-"))

# One-hot
for col in ("position", "team"):
    dummies = pd.get_dummies(players[col], prefix=col)
    players = pd.concat([players, dummies], axis=1)

prob = pulp.LpProblem("FPL", pulp.LpMaximize)
x = pulp.LpVariable.dict("p", players["web_name"], 0, 1, cat=pulp.LpInteger)
c = pulp.LpVariable.dict("c", players["web_name"], 0, 1, cat=pulp.LpInteger)  # captain

# Objective: total points + extra captain points
players["proj_points"] = (
    players["total_points"] / 38 * 38     # rolling projection (placeholder)
)
prob += pulp.lpSum(players.set_index("web_name").loc[p, "proj_points"] * x[p]
                   for p in players["web_name"]) \
      + pulp.lpSum(players.set_index("web_name").loc[p, "proj_points"] * c[p]
                   for p in players["web_name"])

# Budget
prob += pulp.lpSum(players.set_index("web_name").loc[p, "now_cost"] * x[p]
                   for p in players["web_name"]) <= 100

# Formation
prob += pulp.lpSum(players.set_index("web_name").loc[p, "position_GKP"] * x[p]
                   for p in players["web_name"]) == gk
prob += pulp.lpSum(players.set_index("web_name").loc[p, "position_DEF"] * x[p]
                   for p in players["web_name"]) == df
prob += pulp.lpSum(players.set_index("web_name").loc[p, "position_MID"] * x[p]
                   for p in players["web_name"]) == md
prob += pulp.lpSum(players.set_index("web_name").loc[p, "position_FWD"] * x[p]
                   for p in players["web_name"]) == fw

# Max 3 per team
for team in players["team_name_short"].unique():
    prob += pulp.lpSum(players.set_index("web_name").loc[p, f"team_{team}"] * x[p]
                       for p in players["web_name"]) <= 3

# Captain must be in the XI
for p in players["web_name"]:
    prob += c[p] <= x[p]
# Exactly one captain
prob += pulp.lpSum(c[p] for p in players["web_name"]) == 1

prob.solve(pulp.PULP_CBC_CMD(msg=False))
xi = [p for p in players["web_name"] if pulp.value(x[p]) == 1]
captain = [p for p in players["web_name"] if pulp.value(c[p]) == 1][0]
print(f"Captain: {captain}")
print(players[players["web_name"].isin(xi)])
```

This is the form FPL serious players want.

## 8.6 League / entry scraping

```python
# Top 50 of the global league (page 1)
rankings = pb.fpl.get_rankings(page=1)

# Your own entry (find your entry_id on the FPL website URL)
your_picks = pb.fpl.get_entry_picks_by_gameweek(entry_id=123456, gameweek=10)
print(your_picks.columns.tolist())
# team_id, active_chip, event, points, total_points, rank, overall_rank,
# value, bank, event_transfers, event_transfers_cost, points_on_bench,
# auto_sub_1..4, player_pick_0..14, captain_id, vice_captain_id

transfers = pb.fpl.get_entry_transfers(entry_id=123456)
```

This is the foundation of a "Mini-league dashboard" page: track your
rivals, see who they captained, who they transferred in/out.

## 8.7 Captain-pick advisor

The classic advisor combines form, fixture difficulty, and underlying
expected stats:

```python
# scripts/fpl/captain_advisor.py
import penaltyblog as pb
import pandas as pd

players = pb.fpl.get_player_data()
# Form = recent points per game; ICT = influence + creativity + threat
players["advisor_score"] = (
    0.5 * players["form"] +
    0.3 * players["ict_index"] +
    0.2 * players["points_per_game"]
)
top5 = players.nlargest(5, "advisor_score")[
    ["web_name", "team_name_short", "position_short",
     "now_cost", "form", "ict_index", "advisor_score"]
]
print(top5)
```

Wrap in a Streamlit page with a "explain my captain pick" expander that
shows the score breakdown.

## 8.8 Price-change predictor

FPL price changes are a money-making side game. Watch `transfers_balance`
on each player:

```python
import penaltyblog as pb
import pandas as pd

players = pb.fpl.get_player_data()
# Top 20 by net transfers (most-bought players)
players["net_transfers"] = players["transfers_in_event"] - players["transfers_out_event"]
rising = players.nlargest(20, "net_transfers")[
    ["web_name", "now_cost", "net_transfers", "selected_by_percent"]
]
print(rising)
```

A simple threshold on `net_transfers` (often ±100k per hour) is a
decent price-change predictor. Stream this into a "Watchlist" page.

## 8.9 Pitch Oracle integration

**Touches**

- New `fpl/` workspace inside the EPL consumer repo.
- `fetch_player_data.py`, `fetch_player_data_fd.py` — partly replaced.
- New `scripts/fpl/optimal_with_captain.py`, `scripts/fpl/captain_advisor.py`.
- New `data_files/fpl/` directory with `players.parquet`,
  `player_history.parquet`, etc.
- New `streamlit_pages/10_fpl_home.py`.

**Does not touch**

- The core modelling layer.
- Non-EPL consumer repos.

**Migration cost**

- One engineer, ~3 days for the basic integration; ~6 days for the
  full mini-league dashboard.
