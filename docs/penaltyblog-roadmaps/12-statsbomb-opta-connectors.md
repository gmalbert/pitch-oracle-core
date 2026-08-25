# 12 — StatsBomb & Opta Connectors via MatchFlow

`matchflow.contrib.statsbomb` and `matchflow.contrib.opta` give you
streaming access to two of the largest professional event-data sources.
Both attach as `Flow.statsbomb` and `Flow.opta` namespaces; the rest of
the pipeline (`filter`, `select`, `group_by`, etc.) is unchanged.

Pitch Oracle currently has:

- `explore_statsbomb.py` — eager `pd.read_json` over the local
  StatsBomb open-data archive.
- `explore_wyscout.py` — eager `pd.read_json` over the Wyscout S3 bucket.

The roadmap: replace both with lazy `Flow` pipelines.

## 12.1 StatsBomb open-data

The StatsBomb open-data GitHub repo has the layout:

```
data/
  events/<match_id>.json     # one big file per match
  matches/<competition>/<season>.json
  lineups/<match_id>.json
```

Use `Flow.from_glob()` to stream:

```python
from penaltyblog.matchflow import Flow, where_equals

shots = (
    Flow.from_glob("data_files/statsbomb/open-data/data/events/*.json")
        .filter(where_equals("type.name", "Shot"))
        .select("match_id", "team.name", "player.name",
                "shot.statsbomb_xg", "location", "minute",
                "shot.outcome.name")
        .to_pandas()
)
```

### Live StatsBomb API

For non-open data (paid StatsBomb plans):

```python
from penaltyblog.matchflow.statsbomb import from_statsbomb

flow = (
    from_statsbomb(
        username=os.environ["STATSBOMB_USER"],
        password=os.environ["STATSBOMB_PASS"],
        competition_id=11,
        season_id=27,
    )
    .filter(where_equals("type.name", "Shot"))
    .select("team.name", "player.name",
            "shot.statsbomb_xg", "location", "minute")
)
flow.explain(optimize=True)         # show the plan
df = flow.to_pandas()
```

The first node in the plan is the API call. Nothing is fetched until
`.collect()` (or `.to_pandas()` etc.) runs.

### Pagination handling

`from_statsbomb` handles pagination automatically — it walks through
all matches in the competition/season and emits one record per event.

### Schema coercion

StatsBomb IDs are integers but some fields come back as strings. Use
`with_schema`:

```python
flow = (
    Flow.from_glob(".../events/*.json")
        .with_schema({
            "match_id": int,
            "minute":   int,
            "second":   int,
            "shot.statsbomb_xg": float,
        }, strict=False)
)
```

`strict=False` (default) keeps the original value on cast failure.

## 12.2 Opta / Perform

The Opta adapter connects to Opta's REST API. The two main feed types:

- `fixtures` — match schedule and metadata.
- `events` — per-match event stream.

```python
from penaltyblog.matchflow.opta import from_opta

fx_flow = (
    from_opta(
        base_url="https://api.opta.com/v3",
        auth_key=os.environ["OPTA_AUTH_KEY"],
        asset_type="fixtures",
        competition="ENG_PL",
        season=2024,
    )
    .select("match_id", "kickoff", "home_team", "away_team",
            "venue", "competition_round")
)

events_flow = (
    from_opta(
        base_url="https://api.opta.com/v3",
        auth_key=os.environ["OPTA_AUTH_KEY"],
        asset_type="events",
        competition="ENG_PL",
        season=2024,
    )
    .filter(lambda r: 1 <= r.get("minute", 0) <= 90)
)
```

### Filter to a date window

```python
from datetime import datetime

dec_fixtures = (
    fx_flow
    .filter(lambda r: "2024-12" in r["kickoff"])
    .to_pandas()
)
```

Or use a predicate helper if the format is regular:

```python
from penaltyblog.matchflow import where_contains

dec_fixtures = (
    fx_flow
    .filter(where_contains("kickoff", "2024-12"))
    .to_pandas()
)
```

## 12.3 Compose a joined fixture+events view

The killer use case: per-fixture, give me the shots with player and
team metadata:

```python
shots = (
    events_flow
    .filter(lambda r: r.get("type_id") == 15)   # 15 = Shot in Opta F24
    .select("match_id", "team_id", "player_id",
            "minute", "second", "x", "y", "statsbomb_xg")
)

fixtures = (
    fx_flow
    .select("match_id", "home_team", "away_team",
            "kickoff", "venue")
)

joined = (
    shots.join(fixtures, on="match_id", how="left")
         .to_pandas()
)
```

`join` supports `left`, `right`, `inner`, `outer`, `anti`. Suffixes are
configurable.

## 12.4 Write streams to S3

Both `Flow.to_jsonl` and `Flow.to_parquet` accept `storage_options`:

```python
flow.to_parquet(
    "s3://pitch-oracle-data/events/2024/snappy.parquet",
    storage_options={
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
    },
)
```

The same plan runs locally and in Lambda. MatchFlow installs the
required cloud-storage dependency lazily (`s3fs`, `gcsfs`, `adlfs`)
and raises a helpful error if it's missing.

## 12.5 Use case: training an xT model from StatsBomb

Combine [09-expected-threat.md](09-expected-threat.md) with this
roadmap:

```python
from penaltyblog.matchflow import Flow, where_equals
from penaltyblog.xt import XTModel, XTEventSchema

events = (
    Flow.from_glob("data_files/statsbomb/open-data/data/events/*.json")
        .select("type.name", "location", "pass.end_location",
                "player.name", "team.name", "shot.statsbomb_xg",
                "pass.outcome.name")
        .to_pandas()
)

schema = XTEventSchema(
    x="location.0",
    y="location.1",
    event_type="type.name",
    end_x="pass.end_location.0",
    end_y="pass.end_location.1",
    is_success="pass.outcome.name",
    x_range=(0, 120), y_range=(0, 80),
)
# Map success labels: complete passes / goals → True
events["is_success"] = events["pass.outcome.name"].isna()

model = XTModel().fit(events, schema=schema)
model.save("models/xt_v1.npz")
```

This is the same training pipeline as in [09](09-expected-threat.md),
but the data ingestion is now lazy and scales to the full StatsBomb
archive without ever holding it in memory.

## 12.6 Use case: building the "per-fixture summary" artifact

```python
summary = (
    Flow.from_glob(".../events/*.json")
        .group_by("match_id")
        .summary({
            "n_events":  ("count", None),
            "n_shots":   lambda rows: sum(1 for r in rows
                                          if r["type.name"] == "Shot"),
            "n_xg":      lambda rows: sum((r.get("shot", {}) or {}).get("statsbomb_xg", 0.0)
                                          for r in rows),
            "n_passes":  lambda rows: sum(1 for r in rows
                                          if r["type.name"] == "Pass"),
        })
        .to_pandas()
)
```

That summary table feeds the "Match Centre" page header.

## 12.7 Use case: building xT inputs for the ML ensemble

```python
xt_inputs = (
    Flow.from_glob(".../events/*.json")
        .select("player.name", "type.name",
                "location", "pass.end_location",
                "shot.statsbomb_xg", "pass.outcome.name")
        .assign(xT=lambda r: None)         # filled in by XTModel.score
        .to_pandas()
)
```

`assign` returns a `Flow` whose downstream step is a `map`. For very
large datasets, compute xT in chunks:

```python
chunks = []
for chunk_start in range(0, len(events), 50_000):
    chunk = xt_inputs.iloc[chunk_start:chunk_start + 50_000]
    scored_chunk = model.score(chunk, schema=schema)
    chunks.append(scored_chunk)
all_scored = pd.concat(chunks)
```

## 12.8 Pitch Oracle integration

**Touches**

- `explore_statsbomb.py`, `explore_wyscout.py` — rewrite as MatchFlow
  pipelines.
- New `scripts/events/build_xt_training_set.py` — stream events, score
  with `XTModel`.
- New `scripts/events/build_match_summaries.py` — group_by + summary
  per match.

**Does not touch**

- The modelling layer (MatchFlow is a data source, not a model).
- Streamlit pages (data shape preserved).

**Migration cost**

- One engineer, ~4 days including the StatsBomb xT pipeline and the
  per-fixture summary artifact.

## 12.9 Common pitfalls

1. **Don't fetch all of StatsBomb at once.** The full archive is ~3M
   events; even filtered, you want lazy streaming.
2. **Don't trust that provider coordinates are 0–100.** StatsBomb uses
   0–120 × 0–80; Opta uses 0–100 × 0–100. Set `x_range`/`y_range` in
   the schema.
3. **Don't join on string IDs without coercion.** Use `with_schema` to
   coerce IDs to ints before joining.
4. **Don't hardcode the path.** Use `storage_options` to point at S3
   / GCS / Azure.
