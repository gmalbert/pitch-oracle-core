# 10 — Visualizations

`penaltyblog.viz` is a Plotly-based pitch visualisation library. It
handles provider-specific dimensions (StatsBomb, Wyscout, Opta, custom),
themes, layers, orientations, and views. The public surface:

```python
from penaltyblog.viz import Pitch, Theme, PitchDimensions
from penaltyblog.viz import plot_trace, plot_autocorr, plot_posterior, plot_convergence
```

The two Streamlit touchpoints that benefit the most:

- The "Match Centre" page, which currently renders ad-hoc Plotly
  pitches. Replacing them with `Pitch` guarantees consistent dimensions
  across all consumers.
- The "Model Diagnostics" tab, where Bayesian trace plots are shown.

## 10.1 The Pitch class

```python
from penaltyblog.viz import Pitch

pitch = Pitch(
    provider="statsbomb",     # or "opta", "wyscout", "tracab", "custom"
    width=700, height=500,
    theme="minimal",          # or "classic", "dark", etc.
    orientation="horizontal", # or "vertical"
    view="full",              # "full", "left", "right", "top", "bottom"
    title="Arsenal vs Chelsea",
    subtitle="Premier League 2024/25",
    show_axis=False,
    show_legend=True,
)
```

`Pitch(provider=...)` accepts any of the built-in providers or a custom
`PitchDimensions` instance. `theme` accepts a name or a `Theme` instance.

### Show / save

```python
pitch.show()
pitch.save("output/match_centre.png", scale=2.0)
pitch.save("output/match_centre.svg")
```

The `kaleido` package is required for image export.

## 10.2 Plot scatter points

```python
pitch.plot_scatter(
    shots,
    x="x", y="y",
    hover="player.name",
    color="rgba(255, 50, 50, 0.6)",
    size=10,
)
```

`data` can be a `pandas.DataFrame`, a list of dicts, or a `Flow`. The
`x` and `y` arguments accept dot-paths for nested data.

## 10.3 Plot heatmaps

```python
pitch.plot_heatmap(
    actions, x="x", y="y",
    bins=(16, 12), show_colorbar=True,
)
```

`plot_heatmap` is a 2D histogram; for smooth densities use `plot_kde`:

```python
pitch.plot_kde(
    actions, x="x", y="y", grid_size=120,
    colorscale="Hot", opacity=0.6,
)
```

## 10.4 Plot arrows / comets

For directional movements:

```python
pitch.plot_arrows(
    passes, x="x", y="y", x_end="end_x", y_end="end_y",
    hover="player.name",
    color="rgba(50, 50, 200, 0.7)",
)
```

For movement with a fading tail:

```python
pitch.plot_comets(
    carries, x="x", y="y", x_end="end_x", y_end="end_y",
    fade=True, segments=24, width=4,
)
```

## 10.5 Layer management

The `Pitch` class maintains a layer dictionary. You can hide, show, and
reorder layers:

```python
pitch.set_layer_visibility("arrows", visible=False)
pitch.set_layer_order(["scatter", "heatmap", "arrows", "comets"])
pitch.remove_layer("comets")
```

In Streamlit, expose these as toggles so users can build their own view.

## 10.6 Themes

```python
from penaltyblog.viz import Theme

theme = Theme(
    "minimal",
    pitch_color="#0e1117",
    line_color="#ffffff",
    marker_color="#ff6b6b",
    heatmap_colorscale="Viridis",
    font_family="Inter, sans-serif",
)
pitch = Pitch(theme=theme)
```

Or define a custom theme by extending the default:

```python
from penaltyblog.viz import Theme
default = Theme("minimal")
custom = Theme(
    pitch_color="#0c0c10",
    line_color="#f1faee",
    marker_color="#e63946",
    heatmap_colorscale="Plasma",
)
custom.update(default.to_dict())
```

## 10.7 Custom PitchDimensions

For an in-house pitch (e.g. an amateur football analytics feed):

```python
from penaltyblog.viz import PitchDimensions

custom = PitchDimensions(
    length=105, width=68,
    coordinate_origin="bottom_left",
    coordinate_system="opta",     # x: 0..100, y: 0..100
    penalty_area_length=16.5, penalty_area_width=40.32,
    six_yard_length=5.5, six_yard_width=18.32,
    penalty_spot_distance=11.0,
)
pitch = Pitch(provider=custom, theme="minimal")
```

`PitchDimensions` exposes `scaled_shapes(target_length, target_width)`
and `apply_coordinate_scaling(df, x, y)` — the same plumbing the built-in
providers use.

## 10.8 Bayesian diagnostics

```python
from penaltyblog.viz import plot_trace, plot_autocorr, plot_posterior

plot_trace(trace_dict["home_advantage"])
plot_autocorr(trace_dict["home_advantage"])
plot_posterior(trace_dict["rho"])
plot_convergence(trace_dict)        # full trace diagnostics
```

All four return Plotly figures ready for `fig.show()` or for embedding
in Streamlit via `st.plotly_chart(fig, use_container_width=True)`.

## 10.9 Plot the xT surface

```python
from penaltyblog.xt import load_pretrained_xt
model = load_pretrained_xt()
pitch = model.plot()
pitch.show()
```

`XTModel.plot()` adds the xT heatmap as a layer on a freshly-initialised
`Pitch`. Pass in your own pitch if you want a custom theme:

```python
from penaltyblog.viz import Pitch
pitch = Pitch(provider="statsbomb", theme="dark", view="left")
model.plot(pitch=pitch)
```

## 10.10 Embed in Streamlit

```python
# streamlit_pages/04_match_centre.py
import streamlit as st
from penaltyblog.viz import Pitch

st.title("Match Centre")

events = st.session_state["events"]
home = st.session_state["home"]
away = st.session_state["away"]

pitch = Pitch(provider="statsbomb", theme="minimal",
              title=f"{home} vs {away}", width=900, height=600)

view = st.radio("View", ["Heatmap", "Shots", "Passes", "Carries"], horizontal=True)
if view == "Heatmap":
    pitch.plot_heatmap(events, x="x", y="y")
elif view == "Shots":
    shots = events[events["type.name"] == "Shot"]
    pitch.plot_scatter(shots, x="x", y="y", color="rgba(255, 0, 0, 0.6)",
                      hover="player.name", size=12)
elif view == "Passes":
    passes = events[(events["type.name"] == "Pass") &
                    (events["pass.outcome.name"].isna())]
    pitch.plot_arrows(passes, x="x", y="y",
                      x_end="pass.end_location.0", y_end="pass.end_location.1",
                      color="rgba(0, 100, 200, 0.4)")
elif view == "Carries":
    carries = events[events["type.name"] == "Carry"]
    pitch.plot_comets(carries, x="x", y="y",
                      x_end="carry.end_location.0", y_end="carry.end_location.1",
                      fade=True)

st.plotly_chart(pitch.fig, use_container_width=True)
```

## 10.11 Compose multiple providers

Sometimes you need to compare a StatsBomb pitch to a Wyscout pitch in
the same figure. Build two pitches and overlay the layers:

```python
from penaltyblog.viz import Pitch

sb = Pitch(provider="statsbomb", theme="minimal")
ws = Pitch(provider="wyscout", theme="minimal")

# Use one pitch for the canvas, the other for the data coordinates.
pitch = Pitch(provider="statsbomb", theme="minimal")
pitch.plot_scatter(events, x="x", y="y", hover="player.name")
# Add a second pitch's scatter as an annotation layer
pitch._add_layer("extra", ws.plot_scatter(events, x="x", y="y",
                                          color="rgba(0,200,0,0.4)").to_plotly_json())
```

This is hacky; for production use, normalise to one provider first.

## 10.12 Pitch-side: dashboard tiles

Build a multi-pitch dashboard with `st.columns`:

```python
import streamlit as st

cols = st.columns(4)
for i, (label, df) in enumerate([
    ("Shots", shots),
    ("Passes", passes),
    ("Carries", carries),
    ("Pressures", pressures),
]):
    with cols[i]:
        p = Pitch(provider="statsbomb", theme="minimal",
                  title=label, width=300, height=300)
        p.plot_heatmap(df, x="x", y="y")
        st.plotly_chart(p.fig, use_container_width=True)
```

A 4-tile mini-pitch dashboard is a great landing-page widget.

## 10.13 Common pitfalls

1. **Don't plot on the wrong provider.** If your events are StatsBomb
   (0–120 × 0–80) but the pitch is Wyscout (0–100 × 0–100), the
   coordinates won't align.
2. **Don't mix axes between providers.** Convert first.
3. **Don't forget `kaleido` for static export.** `pitch.save()` will
   fail silently if it's missing.
4. **Don't reuse the same `Pitch` instance across pages.** Each page
   should construct its own; the layer dict is mutable.

## 10.14 Pitch Oracle integration

**Touches**

- New `streamlit_components/pitch.py` wrapping `Pitch` with sensible
  defaults.
- Streamlit pages that currently embed ad-hoc Plotly pitches:
  `streamlit_pages/04_match_centre.py`, `06_team_profile.py`,
  `07_expected_threat.py`, `08_player_profile.py`.

**Does not touch**

- The modelling layer.

**Migration cost**

- One engineer, ~4 days including the four-page rewrite.
