# 09 — Expected Threat (xT)

`penaltyblog.xt` is a position-based Expected Threat model. The idea
(analogous to xG, but for *any* on-ball action) is to estimate how much
a given ball action increases the probability that the attacking team
scores before losing possession. The pitch is divided into an
`n_cols × n_rows` grid; the model jointly solves for shot probability,
goal probability, and movement probability per cell.

```python
from penaltyblog.xt import XTModel, XTEventSchema, load_pretrained_xt
```

There are two main entry points:

- `XTModel().fit(df)` — fit a custom model on your own event data.
- `load_pretrained_xt()` — load a ready-to-use pretrained surface.

Both produce an `XTModel` that can `score(df)`, return per-action xT
deltas (`xt_added`), and `plot()` a heatmap on a Plotly pitch.

## 9.1 Load the pretrained model

```python
from penaltyblog.xt import load_pretrained_xt

model = load_pretrained_xt()

# Query a single point
print(model.value_at(85, 50))     # xT near the penalty spot
print(model.value_at(10, 50))     # xT in own half

# Build a full-pitch heatmap
import numpy as np
xs = np.linspace(0, 100, 16)
ys = np.linspace(0, 100, 12)
xx, yy = np.meshgrid(xs, ys)
vals = model.values_at(xx.ravel(), yy.ravel()).reshape(12, 16)

# Plot
pitch = model.plot()
pitch.show()
```

The pretrained model is fitted on multi-season Opta data across several
European leagues. It's a useful default for casual queries.

## 9.2 Fit your own xT model

```python
from penaltyblog.xt import XTModel, XTEventSchema

# Required columns: x, y, event_type, end_x, end_y, is_success
df = pd.read_parquet("data_files/events/all_actions.parquet")

schema = XTEventSchema(
    x="x",
    y="y",
    event_type="event_type",
    end_x="end_x",
    end_y="end_y",
    is_success="is_success",
    x_range=(0, 100),       # adapt to your provider
    y_range=(0, 100),
)

model = XTModel(
    n_cols=16,
    n_rows=12,
    include_carries=True,
    include_throw_ins=True,
    include_goal_kicks=True,
    include_corners=True,
    include_free_kicks=True,
    transition_smoothing_k=5.0,
    coord_policy="warn",    # 'warn' / 'error' / 'clip'
)
model.fit(df, schema=schema)
```

The fit is fast — seconds on a 100k-event dataset. Internally it:

1. Discretises coordinates into the `n_cols × n_rows` grid.
2. Computes shot, move, and goal probability per cell.
3. Builds a per-family transition matrix (passes, carries, throw-ins,
   goal kicks, corners, free kicks), shrunk toward a pooled prior.
4. Solves `(I - MT) X = S` where `S = shot_prob * goal_prob`.

The fitted surface has all the right properties:

- Higher in the final third.
- Higher near the goal.
- Highest just outside the six-yard box.

### Save / load

```python
model.save("models/xt_v1.npz")
loaded = XTModel.load("models/xt_v1.npz")
```

`.npz` is portable across machines and across Python versions.

## 9.3 Score actions

```python
scored = model.score(df, schema=schema)
print(scored[["xt_start", "xt_end", "xt_added"]].describe())
```

`score()` adds three columns to your event DataFrame:

- `xt_start` — xT value at the action's start (set for **all** moves,
  even failed ones — you can measure the value risked).
- `xt_end` — xT value at the action's destination (NaN for failed moves
  and shots).
- `xt_added` — `xt_end - xt_start`.

### Aggregate per-player xT

```python
per_player = (
    scored.groupby("player.name")
          .agg(xt_added_total=("xt_added", "sum"),
               xt_added_per90=("xt_added", lambda s: s.sum() / len(s) * 90),
               successful_actions=("xt_added", "count"))
          .sort_values("xt_added_per90", ascending=False)
)
per_player.head(20)
```

This is the "xT per 90" leaderboard that fans love.

### Aggregate per-team pressing value

```python
# Pressing = tackles + interceptions in opponent half
pressing = scored[scored["event_type"].isin(["Tackle", "Interception"])]
pressing_team = (
    pressing.groupby("team.name")
            .agg(xt_added_total=("xt_added", "sum"),
                 actions=("xt_added", "count"))
            .sort_values("xt_added_total", ascending=False)
)
```

## 9.4 Stream xT scoring with MatchFlow

`XTModel.fit()` and `.score()` both accept a `Flow` as input. Stream
directly from S3 or from disk:

```python
from penaltyblog.matchflow import Flow, where_equals
from penaltyblog.xt import XTModel, XTEventSchema

flow = (
    Flow.from_glob("data_files/statsbomb/open-data/data/events/*.json")
        .filter(where_equals("type.name", "Pass"))
        .to_pandas()
)
schema = XTEventSchema(
    x="location.0", y="location.1",
    end_x="pass.end_location.0", end_y="pass.end_location.1",
    event_type="type.name",
    is_success="pass.outcome.name",
    x_range=(0, 120), y_range=(0, 80),
)
model = XTModel().fit(flow, schema=schema)
```

The coordinate scaling happens automatically via the schema.

## 9.5 Per-action xT explained

For a single pass:

```python
ev = scored.iloc[42]
print(f"Player: {ev['player.name']}")
print(f"Action: {ev['event_type']} from ({ev['x']:.1f}, {ev['y']:.1f}) "
      f"to ({ev['end_x']:.1f}, {ev['end_y']:.1f})")
print(f"xT before: {ev['xt_start']:.4f}")
print(f"xT after:  {ev['xt_end']:.4f}")
print(f"xT added:  {ev['xt_added']:+.4f}")
```

That's the unit of analysis for every "what did this player actually
do" conversation.

## 9.6 Use xT as a model feature

`xt_added_per90` is one of the strongest predictive features you can
add to the ML ensemble:

```python
features = slate.merge(per_player[["xt_added_per90"]],
                       left_on="player.name", right_index=True, how="left")
features["xt_added_per90"] = features["xt_added_per90"].fillna(0)
features.to_parquet("data_files/features/xt_features.parquet")
```

The ensemble then learns "high-xT players move my probability by N%".

## 9.7 Pitch heatmap of xT gained

```python
import pandas as pd
import numpy as np

# All passes that ended in the final third, by player
final_third = scored[(scored["xt_end"].notna()) &
                     (scored["end_x"] >= 66.6)]

pitch = model.plot()
pitch.plot_scatter(
    final_third, x="end_x", y="end_y", hover="player.name",
    color="rgba(255, 100, 100, 0.5)",
)
pitch.show()
```

This is a "where does this player advance the ball into dangerous
areas" view — the kind of visual that gets shared on social media.

## 9.8 Compare your xT model to the pretrained one

A neat diagnostic: compute the rank correlation of `xt_added_per90`
between your model and the pretrained one.

```python
import scipy.stats as st

custom_per_player = (
    scored.groupby("player.name")["xt_added"].sum() / scored.groupby("player.name").size() * 90
)

# Score the same actions with the pretrained model
pretrained_scored = load_pretrained_xt().score(df, schema=schema)
pretrained_per_player = (
    pretrained_scored.groupby("player.name")["xt_added"].sum()
    / pretrained_scored.groupby("player.name").size() * 90
)

joined = pd.concat([custom_per_player.rename("custom"),
                    pretrained_per_player.rename("pretrained")], axis=1).dropna()
print(st.spearmanr(joined["custom"], joined["pretrained"]))
```

A Spearman ρ above 0.7 means your model agrees with the pretrained one
on player rankings; a ρ below 0.3 means one of them is off.

## 9.9 xT as a SQL-friendly table

```python
(
    scored.assign(
        grid_x=lambda d: (d["x"] // 6.25).astype(int).clip(0, 15),
        grid_y=lambda d: (d["y"] // 8.33).astype(int).clip(0, 11),
    )
    .groupby(["team.name", "grid_x", "grid_y"])
    .agg(xt_added=("xt_added", "sum"), n=("xt_added", "count"))
    .reset_index()
    .to_parquet("data_files/xt/cell_team.parquet")
)
```

That table is consumed directly by the Streamlit "Pitch control" page.

## 9.10 Pitch Oracle integration

**Touches**

- New `scripts/xt/fit_xt.py` — fit one xT model per (competition,
  season).
- New `scripts/xt/score_events.py` — apply the model to every event.
- New `data_files/xt/` directory with fitted `.npz` files and scored
  parquet tables.
- New `scripts/ensemble/build_features.py` — add `xt_added_per90` to
  the feature set.
- New `streamlit_pages/07_expected_threat.py` — the consumer.

**Does not touch**

- The goal models.
- Existing ingestion (xT is computed from event data, not fixtures).

**Migration cost**

- One engineer, ~3 days (assuming StatsBomb open data is already
  ingested via [02-matchflow-event-pipelines.md](02-matchflow-event-pipelines.md)).

## 9.11 Common pitfalls

1. **Don't fit on shots alone.** xT needs move events to learn
   transitions; a fit-only-on-shots surface is degenerate.
2. **Don't ignore the `coord_policy`.** If your provider uses 0–120 ×
   0–80 coordinates (StatsBomb) and you don't set `x_range`/`y_range`
   in the schema, you'll get a stretched surface.
3. **Don't treat xT as xG.** xT is a per-action value; xG is a
   per-shot probability. They live on different scales and mean
   different things.
4. **Don't average xT across actions of different types.** A successful
   long pass and a dribble past one opponent both have xT, but they
   mean different things. Segment first.
