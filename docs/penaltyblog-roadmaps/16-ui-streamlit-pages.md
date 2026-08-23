# 16 — Streamlit / UI Consolidation

The roadmaps for the new content (03, 04, 05, 06, 07, 09, 10, 11) each
touch the UI in different places. Without a consolidation, the
Streamlit pages end up with five ad-hoc Plotly pitch snippets, three
different colour schemes, two reliability diagrams, and no shared
header / footer. This roadmap fixes that.

## 16.1 The new information architecture

| Page | Source roadmap | Replaces |
|------|----------------|----------|
| `00_overview.py` | existing + 11 | unchanged, but reads the new grid |
| `01_match_center.py` | 09, 10, 11 | existing match prediction page |
| `02_fixture_detail.py` | 11 | new — every market in one DataFrame |
| `03_value_bets.py` | 06, 07 | new |
| `04_today_bets.py` | 07 | new |
| `05_team_center.py` | 05 | existing team form page |
| `06_team_comparison.py` | 05 | new |
| `07_power_rankings.py` | 05 | new |
| `08_expected_threat.py` | 09 | new |
| `09_fpl_home.py` | 08 | new (EPL only) |
| `10_model_lab.py` | 03, 04, 14 | new |
| `11_market_lab.py` | 06, 07 | new — capability-gated |
| `12_prediction_history.py` | 14 | new — append-only ledger |
| `13_data_control_room.py` | 15 | new — severity tiles |
| `14_calibration.py` | 14 | new — reliability diagrams |

Pages 1–7 are P1; pages 8–10 are P2; pages 11–14 are P3. Page 9 (FPL)
is EPL-only and capability-gated.

## 16.2 Reusable component layer

A single `pitch_oracle_core/ui/components/` directory:

```text
ui/components/
  pitch.py            # wraps pb.viz.Pitch with sensible defaults
  probability.py      # tile for 1X2 / BTTS / O-U
  score_matrix.py     # the joint-distribution heatmap (F02)
  reliability.py      # reliability diagram (F37)
  freshness.py        # "data observed at ..." badge
  drivers.py          # forecast-driver waterfall (F04)
  ratings_table.py    # 4-system leaderboard
  team_trends.py      # Elo / form chart
  projection_table.py # season-simulation table
```

The components own the visual style. Pages consume them.

### `components/pitch.py`

A single wrapper that enforces one theme, one provider, and one
orientation:

```python
# pitch_oracle_core/ui/components/pitch.py
from penaltyblog.viz import Pitch, Theme
import streamlit as st

THEME = Theme(
    "pitch-oracle",
    pitch_color="#0e1117",
    line_color="#f1faee",
    marker_color="#e63946",
    heatmap_colorscale="Plasma",
    font_family="Inter, sans-serif",
)

def pitch(provider="statsbomb", **kwargs) -> Pitch:
    p = Pitch(provider=provider, theme=THEME, show_axis=False,
              show_legend=True, **kwargs)
    return p
```

Every page does `from pitch_oracle_core.ui.components.pitch import
pitch` then `p = pitch("statsbomb", title=...)`. There is exactly
one `Pitch` constructor in the entire app.

### `components/probability.py`

A 1X2 / BTTS / O-U tile that always renders the same way:

```python
# pitch_oracle_core/ui/components/probability.py
import streamlit as st

def prob_tile(label: str, value: float, fmt: str = "{:.1%}",
              delta: float | None = None):
    st.metric(label, fmt.format(value),
              delta=(f"{delta:+.1%}" if delta is not None else None))
```

Pages call `prob_tile("Home win", grid.home_win)`; the visual style
is fixed.

### `components/score_matrix.py`

The joint-distribution heatmap (F02 in the product-expansion catalog):

```python
# pitch_oracle_core/ui/components/score_matrix.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from penaltyblog.models import FootballProbabilityGrid

def score_matrix(g: FootballProbabilityGrid, max_goals: int = 6):
    home_pmf = g.home_goal_distribution()[:max_goals + 1]
    away_pmf = g.away_goal_distribution()[:max_goals + 1]
    mat = np.outer(home_pmf, away_pmf)
    fig = go.Figure(data=go.Heatmap(
        z=mat, x=list(range(max_goals + 1)), y=list(range(max_goals + 1)),
        colorscale="Plasma", text=np.round(mat, 3),
        texttemplate="%{text}", colorbar=dict(title="P"),
    ))
    fig.update_layout(xaxis_title="Away goals", yaxis_title="Home goals",
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
```

## 16.3 Reusable page shells

Three shells are reused across the new pages:

```text
ui/shells/
  match_shell.py     # Match Centre, Fixture Detail, Value Bets
  team_shell.py      # Team Center, Team Comparison, Power Rankings
  model_shell.py     # Model Lab, Calibration, Prediction History
```

Each shell renders the header (title + freshness badge), the
capability flags (hide cards that need a missing provider), the page
body, and a footer with model id + manifest version.

```python
# pitch_oracle_core/ui/shells/match_shell.py
import streamlit as st
from pitch_oracle_core.ui.components.freshness import freshness_badge

def match_shell(home, away, kickoff_utc, body):
    st.title(f"{home} vs {away}")
    freshness_badge(kickoff_utc)
    body()
    _footer()

def _footer():
    from pitch_oracle_core.ui.components.footer import footer
    footer()
```

Pages look like:

```python
# streamlit_pages/02_fixture_detail.py
from pitch_oracle_core.ui.shells.match_shell import match_shell
from pitch_oracle_core.ui.components.score_matrix import score_matrix
from pitch_oracle_core.ui.components.probability import prob_tile
from pitch_oracle_core.markets import market_grid

def body():
    g = load_grid(home, away)
    cols = st.columns(3)
    prob_tile("Home", g.home_win)
    prob_tile("Draw", g.draw)
    prob_tile("Away", g.away_win)
    score_matrix(g)
    st.dataframe(market_grid(g), use_container_width=True)

match_shell(home, away, kickoff_utc, body)
```

There is no `st.set_page_config` inside the page; it lives in
`app_factory.py`.

## 16.4 Freshness and capability badges

Every page shows when its data was last updated. A single
`freshness_badge(observed_at_utc)` component renders the timestamp and
warns if it is older than the configured threshold:

```python
# pitch_oracle_core/ui/components/freshness.py
import streamlit as st
from datetime import datetime, timezone
import pandas as pd

def freshness_badge(observed_at_utc, max_age_hours: int = 24):
    age = (pd.Timestamp.utcnow() - pd.Timestamp(observed_at_utc)).total_seconds() / 3600
    if age < max_age_hours:
        st.caption(f"Data observed {age:.1f}h ago")
    else:
        st.warning(f"Data is {age:.1f}h old; refresh artifacts.")
```

Capability badges are simpler. The manifest v3 carries a
`capabilities` block; pages read it via a single helper:

```python
# pitch_oracle_core/ui/components/capabilities.py
import streamlit as st
from pitch_oracle_core.artifacts.manifest import load_manifest

def has_capability(name: str) -> bool:
    return name in load_manifest().get("capabilities", [])

def require(name: str):
    if not has_capability(name):
        st.info(f"{name} is not available for this consumer.")
        st.stop()
```

Pages do:

```python
require("odds")
require("fpl")
require("events")
```

A consumer without an odds provider renders an honest "not available"
state; the rest of the page works.

## 16.5 Model Lab page

The most complex new page. It has six tabs:

1. **Champion** — current model, parameters, fit diagnostics, gating
   summary.
2. **Challengers** — every other model that has been registered, with
   its paired comparison vs the champion.
3. **Reliability** — per-market reliability diagram + ECE.
4. **Cohorts** — table of cohort metrics, sortable, with min-sample
   badge.
5. **Drift** — severity tiles, PSI top-10, calibration breaches.
6. **Diagnostics artifact** — raw JSON viewer.

The page is a single script that reads
`artifacts/diagnostics.json` and `artifacts/severity.json`. It does
no live compute.

## 16.6 Today / Value Bets pages

The Value Bets page lists every opportunity in the slate, with
filters for:

- Minimum edge (default 2%)
- Kelly fraction (default 0.25)
- Whether to include arbitrage
- Whether to include partial-book coverage

The Today's Bets page is a one-line summary: "N value bets, M
arbitrage opportunities, expected growth X%". It links to the Value
Bets page with the relevant filters pre-applied.

```python
# streamlit_pages/04_today_bets.py
import streamlit as st
import pandas as pd

bets = pd.read_parquet("data_files/value/daily_scan.parquet")
if bets.empty:
    st.info("No opportunities today.")
    st.stop()

arb = bets[bets["type"] == "arbitrage"]
val = bets[bets["type"] == "value"]

c1, c2, c3 = st.columns(3)
c1.metric("Value bets", len(val))
c2.metric("Arbitrage", len(arb))
c3.metric("Total expected growth",
          f"{val['expected_growth'].sum():.2%}")

if st.button("Open Value Bets page"):
    st.switch_page("pages/03_value_bets.py")
```

## 16.7 Theming

The single theme instance (`THEME` in `components/pitch.py`) is the
only place colours are defined. Pages that need additional colours
extend `THEME` locally, never redefine it.

```python
# pitch_oracle_core/ui/theme.py
from pitch_oracle_core.ui.components.pitch import THEME

POSITIVE = "#06d6a0"
NEGATIVE = "#ef476f"
NEUTRAL  = "#ffd166"
```

## 16.8 Page-level testing

Every new page has at least one `AppTest` smoke test:

```python
# tests/test_ui_pages.py
from streamlit.testing.v1 import AppTest

def test_match_center_renders():
    at = AppTest.from_file("streamlit_pages/01_match_center.py")
    at.session_state["match_id"] = "test-fixture"
    at.run()
    assert not at.exception
    assert "Arsenal vs Chelsea" in at.title[0].value

def test_value_bets_handles_empty():
    at = AppTest.from_file("streamlit_pages/04_today_bets.py")
    at.session_state["bets"] = pd.DataFrame()
    at.run()
    assert not at.exception
    assert "No opportunities" in at.markdown[0].value
```

These tests are the bar; a page that cannot be exercised in 30
seconds of AppTest does not ship.

## 16.9 Pitch Oracle integration

**Touches**

- New `pitch_oracle_core/ui/components/` directory.
- New `pitch_oracle_core/ui/shells/` directory.
- All 15 pages in `streamlit_pages/` rewritten or added.
- `app_factory.py` registers the new pages in the order
  [16.1](#161-the-new-information-architecture).
- New `tests/test_ui_pages.py`.

**Does not touch**

- The modelling layer; pages are pure consumers of artifacts.

**Migration cost**

- One engineer, ~6 days including the Model Lab page, the
  reliability component, and the freshness/capability badges.

## 16.10 Anti-patterns

1. **Don't write `st.set_page_config` inside a page.** It belongs in
   `app_factory.py`.
2. **Don't `import streamlit as st` inside a component.** Components
   return figures / data; pages wrap them in `st.plotly_chart(...)`.
3. **Don't recompute on every rerun.** Use `@st.cache_resource` for
   model loads and `@st.cache_data` for parquet reads.
4. **Don't use absolute paths.** Every artifact is referenced through
   the manifest.
5. **Don't hard-code a team name in copy.** Templates use `{{ home }}`
   / `{{ away }}`.
