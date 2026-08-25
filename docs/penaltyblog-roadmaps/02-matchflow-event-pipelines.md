# 02 — MatchFlow Event Pipelines

`penaltyblog.matchflow` is a lazy streaming pipeline for nested JSON.
It is the answer to two scripts in `pitch-oracle-core/` that already exist
and have serious scaling problems:

- `explore_statsbomb.py` — loads the StatsBomb open-data archive into
  memory as a single `pandas.DataFrame`. A full La Liga season is fine
  (~150k events). The whole archive is ~3M events and explodes RAM.
- `explore_wyscout.py` — same pattern, with Wyscout event payloads.

`MatchFlow` exposes a small, fluent API:

```
from_json  -> filter / assign / select / flatten / explode / join
           -> group_by / summary / sort_by / limit / drop / distinct
           -> cache / collect / to_pandas / to_jsonl / to_parquet
```

Every step is a plan node. Nothing is executed until `.collect()` is
called, and `FlowOptimizer` rewrites the plan before execution so filters
push down, projections push down, and joins are reordered. The same plan
can stream from local disk, from S3, or from the StatsBomb API.

## 2.1 Stream the StatsBomb open-data archive

The StatsBomb open-data repo has this layout:

```
data/
  events/
    15986.json        # 2022 World Cup final
    ...
  matches/
    11/3.json         # competition_id/season_id.json (list of matches)
    ...
  lineups/
    15986.json
    ...
```

A naive approach loads all of it. MatchFlow lets you stream:

```python
# scripts/events/stream_statsbomb.py
from pathlib import Path
from penaltyblog.matchflow import Flow, where_equals, where_gt

ROOT = Path("data_files/statsbomb/open-data/data")

# 1) start from the events/ folder; one .json per match
events = (
    Flow.from_glob(f"{ROOT}/events/*.json")
        .select("match_id", "team", "player", "type", "minute",
                "second", "location", "pass", "shot", "carry")
        .filter(where_gt("minute", 0))
)

# 2) collect once into pandas — only events that survived the filter
df = events.to_pandas()
```

`to_pandas()` materialises the stream, but the filter was pushed down so
the read is selective. For interactive work, use `.show(n=5)` to inspect
without materialising, and `.count()` to size the stream.

### Counter-example: today's `explore_statsbomb.py`

```python
# pitch-oracle-core/explore_statsbomb.py — what we have today
import json, glob, pandas as pd

rows = []
for path in glob.glob("data_files/statsbomb/open-data/data/events/*.json"):
    with open(path) as f:
        match = json.load(f)
    for ev in match:
        ev["match_id"] = Path(path).stem
        rows.append(ev)
df = pd.DataFrame(rows)
```

That loop holds everything in memory and pays a per-row Python overhead.
MatchFlow does the same in C for the heavy lifting and only walks Python
for user-supplied predicates.

## 2.2 Stream directly from the StatsBomb API

`matchflow.contrib.statsbomb` provides a `from_statsbomb()` source that
calls the public StatsBomb API. It returns a `Flow` whose first node is
the API call; nothing is fetched until `.collect()` runs.

```python
from penaltyblog.matchflow.statsbomb import from_statsbomb

flow = (
    from_statsbomb(
        username=os.environ["STATSBOMB_USER"],
        password=os.environ["STATSBOMB_PASS"],
        competition_id=11,      # La Liga
        season_id=27,           # 2018/19
    )
    .filter(where_equals("type.name", "Shot"))
    .select("team.name", "player.name", "shot.statsbomb_xg", "location")
    .to_pandas()
)
```

Two practical notes:

1. **Credentials** — StatsBomb requires a free account. Store them in
   `.env` and never in the repo.
2. **Rate limits** — the adapter paginates automatically, but be polite.

## 2.3 Stream Opta / Wyscout feeds

Opta feeds are the format used by most European leagues. MatchFlow has an
Opta adapter too:

```python
from penaltyblog.matchflow.opta import from_opta

flow = (
    from_opta(
        base_url="https://api.opta.com/v3",
        auth_key=os.environ["OPTA_AUTH_KEY"],
        asset_type="fixtures",
        competition="ENG_PL",
        season=2024,
    )
    .select("match_id", "kickoff", "home_team", "away_team",
            "venue", "competition_round")
    .filter(lambda r: r["kickoff"].startswith("2024-12"))
    .to_jsonl("data_files/opta/dec_fixtures.jsonl")
)
```

Write the streamed result to JSONL, Parquet, or directly to S3 with
`storage_options={"key": ..., "secret": ...}`. The same plan can run
locally or in a Lambda with no code changes.

### Counter-example: today's `explore_wyscout.py`

```python
# pitch-oracle-core/explore_wyscout.py — what we have today
import pandas as pd, s3fs
fs = s3fs.S3FileSystem(anon=True)
keys = fs.ls("s3://wyscout-open-data/events/England/")
dfs = [pd.read_json(fs.open(k)) for k in keys]
df = pd.concat(dfs)
```

That concatenates everything into RAM. MatchFlow does not:

```python
from penaltyblog.matchflow import Flow

flow = (
    Flow.from_glob("s3://wyscout-open-data/events/England/*.json",
                   storage_options={"anon": True})
        .filter(where_equals("eventName", "shot"))
        .select("playerId", "matchId", "eventSec", "positions",
                "shot.xG", "shot.isGoal")
        .to_parquet("data_files/wyscout/england_shots.parquet")
)
```

The `to_parquet` sink writes incrementally, so memory stays bounded.

## 2.4 Compose complex pipelines

MatchFlow's killer feature is that the plan is a value — you can
compose, optimise, and serialise it. Example: build a per-team pressing
intensity report:

```python
# analysis/pressing/team_pressing_intensity.py
from penaltyblog.matchflow import (
    Flow, where_equals, where_gt, predicates,
)

events = (
    Flow.from_glob("data_files/statsbomb/open-data/data/events/*.json")
        .select("team.name", "type.name", "duel.type",
                "match_id", "minute", "second", "location")
)

# Filter to defending-half pressing duels
defending_press = (
    events
    .filter(where_equals("type.name", "Duel"))
    .filter(where_equals("duel.type", "Tackle"))
    .filter(lambda r: r["location"][0] < 60.0)   # StatsBomb coordinates 0..120
    .assign(minute=lambda r: r["minute"] + r["second"] / 60.0)
)

# Group by team and per-match
per_team_match = (
    defending_press
    .group_by("team.name", "match_id")
    .summary({
        "press_count":  ("count", None),
        "avg_minute":   ("avg", "minute"),
    })
    .to_pandas()
)
```

Note: `summary` accepts either a callable or a dict of `{alias:
(callable_or_name, field)}` — exactly the API the `.matchflow.recipes`
docs describe.

## 2.5 Live join: events + lineups + matches

A common request is "show me every shot by player X with their position
context":

```python
from penaltyblog.matchflow import Flow, where_equals

shots = (
    Flow.from_glob("data_files/statsbomb/open-data/data/events/*.json")
        .filter(where_equals("type.name", "Shot"))
        .select("match_id", "team.name", "player.name", "player.id",
                "shot.statsbomb_xg", "location", "minute")
        .flatten()
)

lineups = (
    Flow.from_glob("data_files/statsbomb/open-data/data/lineups/*.json")
        .select("match_id", "team_id", "team_name", "player_id",
                "player_name", "position")
        .flatten()
)

joined = (
    shots.join(lineups,
               left_on="player.id", right_on="player_id",
               how="left")
         .select("team_name", "player_name", "position",
                 "shot.statsbomb_xg", "location", "minute")
         .to_pandas()
)
```

`join` supports `left`, `right`, `inner`, `outer`, `anti`. Suffixes are
configurable. Type coercion across join keys is configurable (`strict`,
`auto`, `string`).

## 2.6 Use MatchFlow for non-football JSON too

The pipeline is provider-agnostic. Some Pitch Oracle consumers ingest
weather JSON from OpenWeather, injury JSON from a custom scraper, or
even Wikipedia match reports. All of those flow through MatchFlow:

```python
from penaltyblog.matchflow import Flow, where_contains

wiki_reports = (
    Flow.from_jsonl("data_files/wiki_match_reports.jsonl")
        .filter(where_contains("text", "red card"))
        .select("title", "url", "text")
        .limit(50)
        .to_pandas()
)
```

That `limit` pushes down: only the first 50 matching records are ever
read.

## 2.7 Profile and optimise

Two debugging tools are first-class:

```python
flow.explain(optimize=True)        # human-readable plan
flow.explain(optimize=True, compare=True)  # raw vs optimised side-by-side
flow.plot_plan(compare=True)       # matplotlib DAG diagram
flow.profile(fmt="table")          # per-step timing + row counts
```

For a StatsBomb ingest, the plan typically collapses 60+ nodes to
~12 after `FlowOptimizer` runs, because projections and filters are
pushed below group-bys.

## 2.8 Custom UDFs vs predicates

MatchFlow distinguishes between:

- **Predicates** (compiled): `where_equals`, `where_gt`, `where_in`,
  `where_contains`, `where_exists`, `where_is_null`, plus `and_`, `or_`,
  `not_`. Use these for filters that the executor can push down.
- **UDFs** (Python lambda): `flow.filter(lambda r: r["x"] > 0)`. Slower,
  but flexible. Use them sparingly, and only after a predicate filter
  has narrowed the row count.

Rule of thumb: **always** use predicates for filter chains, **only** use
lambdas for `assign` (because the new value depends on old values).

```python
from penaltyblog.matchflow import where_gt, where_equals

flow = (
    Flow.from_glob("events/*.json")
        .filter(where_gt("minute", 0))                # predicate — pushdown
        .filter(where_equals("type.name", "Shot"))     # predicate — pushdown
        .assign(xg=lambda r: r.get("shot", {}).get("statsbomb_xg", 0.0))  # UDF
        .select("team.name", "player.name", "xg")
)
```

## 2.9 Caching and materialisation

For a sub-pipeline that gets reused (e.g., "all shots in the box"),
`.cache()` materialises it once and reuses the result downstream:

```python
shots_in_box = (
    Flow.from_glob("events/*.json")
        .filter(where_equals("type.name", "Shot"))
        .filter(lambda r: 102.0 <= r["location"][0] <= 120.0
                          and 18.0  <= r["location"][1] <= 62.0)
        .cache()
)

# 50 different downstream analyses without re-reading events
finishing_summary = shots_in_box.summary({"xg": ("sum", "shot.statsbomb_xg")})
finishing_summary.to_pandas()
```

`cache()` returns a new `Flow` whose first node is `from_materialized`.
It is mutable from the caller's perspective only via the optimiser.

## 2.10 Schema inference and typing

```python
flow.schema()         # dict[str, type]
flow.keys(limit=100)  # set[str]
```

Use `.with_schema({...}, strict=False, drop_extra=True)` to enforce a
schema, coerce types, and prune unwanted fields. Strict mode raises on
cast failure; the default falls back to the original value. This is
particularly useful when one provider's event JSON has integer IDs and
another has string IDs.

## 2.11 Pitch Oracle integration

**Touches**

- `explore_statsbomb.py`, `explore_wyscout.py` — rewrite as thin
  MatchFlow pipelines.
- `precompute_database.py` — replace the JSON-loading step with a
  MatchFlow `from_glob` that materialises once per season.
- New `scripts/events/matchflow_*.py` family for re-usable event
  pipelines (shots, passes, xT actions, pressing duels).
- New `tests/test_matchflow_pipelines.py` that asserts deterministic
  output across runs.

**Does not touch**

- The Streamlit app (data shape unchanged).
- The modelling layer (MatchFlow sits between ingestion and the
  `models` module).

**Migration cost**

- One engineer, ~4 days. The largest risk is that `explore_wyscout.py`
  may have inlined business logic that has to be re-expressed in
  MatchFlow predicates.
