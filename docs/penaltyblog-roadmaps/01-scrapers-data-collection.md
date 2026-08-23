# 01 — Scrapers & Data Collection

The `penaltyblog.scrapers` module is the easiest entry point for Pitch
Oracle. It contains four scrapers that, between them, cover every
hand-written fetch script currently living in `pitch-oracle-core/`:

| `penaltyblog` scraper | Replaces in Pitch Oracle            | Coverage                                  |
|-----------------------|--------------------------------------|-------------------------------------------|
| `FootballData`        | historical CSV pull (manual)        | 11+ leagues across Europe                 |
| `FBRef`               | any ad-hoc player/squad page scrape  | Big-5 + many more (rate-limited 3s)       |
| `Understat`           | `fetch_understat_xg.py`              | EPL, La Liga, Bundesliga, Serie A, Ligue 1, RPL |
| `ClubElo`             | `fetch_clubelo.py`                   | All teams, daily Elo                      |

All four return a normalised `pandas.DataFrame` with the same column
conventions (`team_home`, `team_away`, `goals_home`, `goals_away`, `xg_home`,
`xg_away`, `date`, `datetime`, `competition`, `season`) and an indexed
`id` column that is stable across scrapers. That alone is worth the
migration: today Pitch Oracle has to maintain `team_name_mapping.py` to
fuse FBRef + Understat + ClubElo + football-data into a single
canonical team identity, and that mapping is incomplete for the Bundesliga
second division.

## 1.1 Replace `fetch_clubelo.py` with `ClubElo`

### Before (current behaviour)

```python
# fetch_clubelo.py — pull today's Elo ratings
import io, requests
import pandas as pd

URL = "http://api.clubelo.com/"
TODAY = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

resp = requests.get(f"{URL}{TODAY}")
elo = pd.read_csv(io.StringIO(resp.text))
elo = elo.rename(columns={"Club": "team"})
elo = elo.sort_values("Elo", ascending=False)
elo.to_csv("data_files/clubelo_today.csv", index=False)
```

The CSV writes the raw club name, which then has to be reconciled against
the canonical names in `team_name_mapping.py`.

### After (penaltyblog)

```python
# fetch_clubelo.py — pull today's Elo ratings with team-name normalisation
from penaltyblog import ClubElo

elo = ClubElo(team_mappings=TEAM_NAME_MAPPINGS).get_elo_by_date()
elo.to_csv("data_files/clubelo_today.csv")
```

The result is a `pandas.DataFrame` indexed by `team`, already snake-cased,
already mapped onto the canonical names. No additional reconciliation step.

### Bonus: full time-series per team

```python
# analysis/diagnostics/elo_trajectory.py
from penaltyblog import ClubElo

ce = ClubElo(team_mappings=TEAM_NAME_MAPPINGS)
arsenal_history = (
    ce.get_elo_by_team("Arsenal")
      .reset_index()
      .rename(columns={"from": "date"})
      [["date", "team", "elo", "rank"]]
)
arsenal_history["elo_28d"] = arsenal_history["elo"].rolling(28, min_periods=1).mean()
arsenal_history["delta_28d"] = arsenal_history["elo"] - arsenal_history["elo"].shift(28)
arsenal_history.to_parquet("data_files/elo/arsenal.parquet")
```

That output feeds the team-form dashboard on the Streamlit page. Because
`ClubElo.get_elo_by_team()` accepts any team string the upstream API
recognises (no `team_mappings` lookup), it's the canonical way to back-fill
multi-season Elo curves for "compare two clubs" views.

## 1.2 Replace `fetch_understat_xg.py` with `Understat`

### Before

```python
# fetch_understat_xg.py — partial snippet from pitch-oracle-core
import json, re, requests
import pandas as pd

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 ..."})

LEAGUES = {"EPL": "EPL", "La Liga": "La_Liga", ...}

def fetch_fixtures(league: str, season: int) -> pd.DataFrame:
    url = f"https://understat.com/getLeagueData/{LEAGUES[league]}/{season}"
    r = session.get(url)
    data = r.json()
    rows = []
    for e in data["dates"]:
        if not e["isResult"]:
            continue
        rows.append({
            "id": e["id"],
            "datetime": e["datetime"],
            "team_home": e["h"]["title"],
            "team_away": e["a"]["title"],
            "goals_home": int(e["goals"]["h"]),
            "goals_away": int(e["goals"]["a"]),
            "xg_home": float(e["xG"]["h"]),
            "xg_away": float(e["xG"]["a"]),
            "forecast_w": float(e["forecast"]["w"]),
            "forecast_d": float(e["forecast"]["d"]),
            "forecast_l": float(e["forecast"]["l"]),
        })
    return pd.DataFrame(rows)
```

This is fine, but it has four known bugs:

1. Headers must include `X-Requested-With: XMLHttpRequest` and a
   `Referer`, otherwise Understat returns HTML.
2. The `cookies={"beget": "begetok"}` trick is undocumented.
3. The `forecast_w/d/l` columns are silently dropped downstream.
4. `team_home`/`team_away` are never normalised.

### After

```python
# fetch_understat_xg.py — penaltyblog does all of the above
from penaltyblog import Understat

scraper = Understat("ENG Premier League", "2024-2025", team_mappings=TEAM_NAME_MAPPINGS)
fixtures = scraper.get_fixtures()                  # full season, normalised
shots   = scraper.get_shots(fixtures.index[0])     # shot-level events
info    = scraper.get_fixture_info(fixtures.index[0])
```

### Bonus: xG deltas by team

```python
# analysis/xg/team_xg_delta.py
from penaltyblog import Understat
import pandas as pd

def team_xg_delta(competition: str, season: str) -> pd.DataFrame:
    fx = Understat(competition, season).get_fixtures()
    home = fx.groupby("team_home")[["goals_home", "xg_home"]].sum()
    away = fx.groupby("team_away")[["goals_away", "xg_away"]].sum()
    out = pd.concat([home, away], axis=1).fillna(0.0)
    out["xg_total"]      = out["xg_home"] + out["xg_away"]
    out["goals_total"]   = out["goals_home"] + out["goals_away"]
    out["xg_delta"]      = out["goals_total"] - out["xg_total"]
    out["xg_delta_pg"]   = out["xg_delta"] / (
        (fx["team_home"].value_counts() + fx["team_away"].value_counts()).reindex(out.index).fillna(0)
    )
    return out.sort_values("xg_delta_pg", ascending=False)

team_xg_delta("ENG Premier League", "2024-2025").to_parquet("data_files/xg_delta/epl_2425.parquet")
```

That dataframe powers the "luck-adjusted table" view.

## 1.3 Replace ad-hoc `football-data.co.uk` pull with `FootballData`

### Before

Today `combine_raw_data.py` does this for one league at a time and stores
the result with custom column names. There is no rate limiting and no
type coercion, so dates come back as strings in two different formats.

### After

```python
# scripts/ingest/football_data_history.py
from penaltyblog import FootballData
from concurrent.futures import ThreadPoolExecutor

LEAGUES = [
    ("ENG Premier League",   "2024-2025"),
    ("ENG Premier League",   "2023-2024"),
    ("ENG Championship",     "2024-2025"),
    ("DEU Bundesliga 1",     "2024-2025"),
    ("ESP La Liga",          "2024-2025"),
    ("ITA Serie A",          "2024-2025"),
    ("FRA Ligue 1",          "2024-2025"),
]

def fetch(league, season):
    fd = FootballData(league, season, team_mappings=TEAM_NAME_MAPPINGS)
    df = fd.get_fixtures()
    df["source"] = "football-data.co.uk"
    return df

with ThreadPoolExecutor(max_workers=4) as ex:
    frames = list(ex.map(lambda p: fetch(*p), LEAGUES))

import pandas as pd
all_fixtures = pd.concat(frames).sort_index()
all_fixtures.to_parquet("data_files/raw/fixtures.parquet")
```

The output is a single, normalised parquet file that every downstream
script already understands.

## 1.4 Replace custom FBRef page parsing with `FBRef`

`FBRef` is the trickiest scraper because FBRef aggressively rate-limits and
returns commented HTML. `penaltyblog` already handles that:

```python
from penaltyblog import FBRef

fbref = FBRef("ENG Premier League", "2024-2025", team_mappings=TEAM_NAME_MAPPINGS)

fixtures = fbref.get_fixtures()
print(fbref.list_stat_types())
# ['standard', 'goalkeeping', 'advanced_goalkeeping', 'shooting',
#  'passing', 'passing_types', 'goal_shot_creation', 'defensive_actions',
#  'possession', 'playing_time', 'misc']

standard  = fbref.get_stats("standard")            # squad_for / squad_against / players
defensive = fbref.get_stats("defensive_actions")
player_xg = fbref.get_stats("shooting")
```

### Creative use: blend FBRef stats with model residuals

```python
# analysis/xg/fbref_xg_residual.py
from penaltyblog import FBRef
import pandas as pd

fbref = FBRef("ESP La Liga", "2024-2025", team_mappings=TEAM_NAME_MAPPINGS)
stats = fbref.get_stats("shooting")["players"]
stats = stats.rename(columns={"expected_xg": "fb_xg"})
stats = stats[["player", "fb_xg", "shots", "shots_on_target"]]

# join onto Understat shots for the same season
from penaltyblog import Understat
shots = Understat("ESP La Liga", "2024-2025").get_shots(
    fixture_id  # populate per-fixture via get_fixtures()
)
# ...
```

This produces a **per-player residual chart** that surfaces systematic FBRef
vs Understat xG disagreements and is great content for the "model
diagnostics" Streamlit tab.

## 1.5 Centralised team-name mapping

The killer feature is that every scraper accepts the same
`team_mappings` dict. Build it once, in `pitch_oracle_core/team_mappings.py`,
and share it everywhere:

```python
# pitch_oracle_core/team_mappings.py
from penaltyblog.scrapers import get_example_team_name_mappings

CANONICAL_TEAM_MAPPINGS = {
    # FBRef / Understat / ClubElo  ->  canonical display name
    "Man United": ["Manchester United"],
    "Man City":   ["Manchester City"],
    "Spurs":      ["Tottenham Hotspur"],
    "Wolves":     ["Wolverhampton Wanderers"],
    "Bayern":     ["Bayern Munich"],
    "Bayer":      ["Bayer Leverkusen"],
    "Atleti":     ["Atletico Madrid"],
    "Inter":      ["Internazionale"],
    "Juve":       ["Juventus"],
    "PSG":        ["Paris Saint-Germain"],
    # ... league-by-league
}

# Re-export the upstream examples so any consumer repo can `from ... import
# CANONICAL_TEAM_MAPPINGS` and get a sensible starting point.
EXAMPLE_MAPPINGS = get_example_team_name_mappings()
```

Then every fetch script becomes:

```python
from pitch_oracle_core.team_mappings import CANONICAL_TEAM_MAPPINGS
from penaltyblog import Understat

scraper = Understat("ENG Premier League", "2024-2025",
                    team_mappings=CANONICAL_TEAM_MAPPINGS)
```

That removes ~250 lines of bespoke name-mangling code from `team_name_mapping.py`.

## 1.6 Rate limiting, retries, and storage

`FBRef` already self-limits (3s gap). `Understat`, `FootballData`, and
`ClubElo` are stateless HTTP, but Pitch Oracle's GitHub Actions should
still wrap them in retries:

```python
# scripts/ingest/_common.py
import time, random, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def session_with_retries() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5, backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10))
    s.headers.update({
        "User-Agent": "pitch-oracle-core/1.3 (+https://pitch-oracle.com)",
        "Accept-Language": "en-GB,en;q=0.9",
    })
    return s
```

Use it as `ClubElo(session=session_with_retries())` if the constructor
accepts one (it does for the underlying `RequestsScraper`).

## 1.7 Migration checklist

- [ ] Replace `fetch_clubelo.py` body with `ClubElo(...).get_elo_by_date()`.
- [ ] Replace `fetch_understat_xg.py` body with `Understat(...).get_fixtures()`.
- [ ] Add `scripts/ingest/football_data_history.py` to back-fill historicals
      across the configured leagues.
- [ ] Add `scripts/ingest/fbref_team_stats.py` to download all stat types
      for each active league once per season.
- [ ] Centralise `TEAM_NAME_MAPPINGS` in `pitch_oracle_core/team_mappings.py`.
- [ ] Remove `team_name_mapping.py` from the root (replaced by the module above).
- [ ] Add `tests/test_penaltyblog_scrapers.py` that asserts the expected
      columns are present in every scraper's output.

## 1.8 Creative extensions

### "Match week snapshot" generator

```python
# scripts/ingest/matchweek_snapshot.py
from penaltyblog import Understat, ClubElo
import pandas as pd
from datetime import datetime

def current_matchweek(competition: str) -> pd.DataFrame:
    fx = Understat(competition, "2024-2025").get_fixtures()
    today = pd.Timestamp.utcnow().normalize()
    fx = fx[fx["datetime"] >= today].sort_values("datetime").head(10)
    elo = ClubElo(team_mappings=TEAM_NAME_MAPPINGS).get_elo_by_date()
    fx = fx.merge(elo[["elo"]], left_on="team_home", right_index=True, how="left") \
           .rename(columns={"elo": "home_elo"}) \
           .merge(elo[["elo"]], left_on="team_away", right_index=True, how="left") \
           .rename(columns={"elo": "away_elo"})
    fx["elo_diff"] = fx["home_elo"] - fx["away_elo"]
    return fx
```

The result is a ten-fixture "this week's slate" view that lands in the
landing-page Streamlit tab.

### "Last 5 by team" form table

```python
# scripts/ingest/recent_form.py
from penaltyblog import FootballData
import pandas as pd

def recent_form(competition: str, season: str, n: int = 5) -> pd.DataFrame:
    fx = FootballData(competition, season).get_fixtures()
    fx = fx.sort_values("datetime", ascending=False)
    rows = []
    for team, group in pd.concat([
            fx[["datetime", "team_home", "goals_home", "goals_away"]].rename(columns={"team_home": "team", "goals_home": "gf", "goals_away": "ga"}),
            fx[["datetime", "team_away", "goals_away", "goals_home"]].rename(columns={"team_away": "team", "goals_away": "gf", "goals_home": "ga"}),
    ]).groupby("team"):
        recent = group.head(n)
        rows.append({
            "team": team,
            "matches": len(recent),
            "gf": recent["gf"].sum(),
            "ga": recent["ga"].sum(),
            "gd": recent["gf"].sum() - recent["ga"].sum(),
            "points": ((recent["gf"] > recent["ga"]) * 3
                       + (recent["gf"] == recent["ga"]) * 1).sum(),
        })
    return pd.DataFrame(rows).sort_values(["points", "gd"], ascending=False)
```

Plug it into `analyze_team_form.py` as a new feature.

### Pitch-side shootout watch

```python
# analysis/diagnostics/xg_vs_elo.py
from penaltyblog import Understat, ClubElo
import matplotlib.pyplot as plt

def xg_vs_elo_scatter(competition: str, season: str):
    fx = Understat(competition, season).get_fixtures()
    elo = ClubElo().get_elo_by_date()
    by_team = fx.groupby("team_home")[["goals_home", "xg_home"]].mean()
    by_team["goals_above_xg"] = by_team["goals_home"] - by_team["xg_home"]
    by_team = by_team.merge(elo[["elo"]], left_index=True, right_index=True)
    by_team.plot.scatter(x="elo", y="goals_above_xg", figsize=(8, 6))
    plt.title(f"{competition} — Elo vs finishing overperformance")
    plt.savefig("output/xg_vs_elo.png", dpi=150)
```

This is a one-off plot, but it generates real conversation in the
weekly newsletter.

## 1.9 Anti-patterns to avoid

1. **Don't double-parse.** If `penaltyblog.scrapers.FBRef` already returns a
   clean DataFrame, do not re-run it through `pd.read_html`. You will lose
   the index, the `id` column, and the normalised names.
2. **Don't strip the `id` column.** It's stable across scrapers and is the
   join key in the canonical fixtures table.
3. **Don't override the `datetime` column with a string.** It's a real
   `pandas.Timestamp` and downstream dashboards depend on that.
4. **Don't cache by URL.** FBRef URLs change slug casing and the API often
   returns 200 with stale content. Cache by `(competition, season, stat_type)` instead.

## 1.10 Pitch Oracle integration

**Touches**

- `fetch_clubelo.py`, `fetch_understat_xg.py`, `team_name_mapping.py` —
  rewrites.
- `combine_raw_data.py` — can become a thin orchestrator that calls the four
  scrapers and concatenates the output.
- `pitch_oracle_core/team_mappings.py` — new shared module.
- `tests/test_scrapers.py` — new test that asserts each scraper produces
  the contract columns.

**Does not touch**

- Streamlit pages. The data shape is identical.

**Migration cost**

- One engineer, ~3 days including tests and retries.
