# UI implementation

The UI should become an analysis product, not a collection of wider tables. Keep
Streamlit and the thin `run_app(config)` entrypoint; replace the monolithic
`ui_pages.py` with page modules, reusable visual components, and an artifact
repository.

## Information architecture

```text
Overview
├── Matchday pulse          F08–F10, F29, F32
├── Race snapshot           F27–F30
└── Data freshness          F47

Match Center
├── Fixture explorer        F01–F07, F11
├── Upset & Draw Radar      F08–F10
└── Prediction history      F38

Teams
├── Team Command Center     F13–F16, F19–F25
└── Comparison Studio       F17–F18

League
├── Live table              F26
├── Season projections      F27–F31
└── League laboratory       F32–F35

Models & Data
├── Model Lab               F36–F45
├── Data Control Room       F46–F48
└── Market Lab              F49–F50 (capability-gated)
```

The default overview should answer “what matters today?” rather than report file and
row counts. Raw files remain downloadable from the Data Control Room, not top-level
navigation.

## Match Center wireframe

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Club A vs Club B · Sat 15:00 local · Stadium · Updated 27m ago             │
│ FULL HISTORY  ·  MODEL dc-elo-stack-v4  ·  FORECAST STABLE                  │
├──────────────────────┬───────────────────────────┬───────────────────────────┤
│ HOME 47% [41–53]     │ DRAW 27% [24–31]         │ AWAY 26% [21–31]         │
├──────────────────────┴───────────────────────────┴───────────────────────────┤
│ Why: + home strength · + opponent away defense · – short rest               │
├───────────────────────────────┬──────────────────────────────────────────────┤
│ Scoreline heatmap             │ Goal outlook                                 │
│ modal: 1–0 / 1–1              │ xG 1.52–1.03 · O2.5 44% · BTTS 48%          │
├───────────────────────────────┼──────────────────────────────────────────────┤
│ Team form sparklines          │ Season stakes                                │
│ opponent-adjusted, last 10    │ win: Europe +8pp · loss: safety –5pp         │
├───────────────────────────────┴──────────────────────────────────────────────┤
│ Context: rest · travel · availability · weather · referee (as available)    │
└──────────────────────────────────────────────────────────────────────────────┘
```

Mobile order is header → probabilities → evidence → score/markets → context →
season stakes. Wide-screen columns are an enhancement, not a dependency.

## Proposed UI package

```text
pitch_oracle_core/ui/
  app.py
  context.py
  repository.py
  navigation.py
  formatters.py
  components/
    probability.py
    score_matrix.py
    drivers.py
    freshness.py
    team_trends.py
    projection_table.py
    reliability.py
  pages/
    overview.py
    match_center.py
    radars.py
    prediction_history.py
    team_center.py
    comparison.py
    standings.py
    projections.py
    league_lab.py
    model_lab.py
    data_control.py
    market_lab.py
```

## 1. Artifact repository and app context

Pages should not know filenames or delimiters.

```python
# pitch_oracle_core/ui/repository.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def _read_parquet(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _read_json(path: str, modified_ns: int) -> dict:
    del modified_ns
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ArtifactRepository:
    root: Path
    descriptors: dict[str, dict]

    @classmethod
    def from_manifest(cls, root: str | Path) -> "ArtifactRepository":
        root = Path(root)
        manifest = json.loads(
            (root / "precomputed" / "cache_manifest.json").read_text(encoding="utf-8")
        )
        artifacts = manifest["artifacts"]
        if isinstance(artifacts, list):
            descriptors = {item["name"]: item for item in artifacts}
        else:  # v2 compatibility during migration
            descriptors = {
                name: {"name": name, **item} for name, item in artifacts.items()
            }
        return cls(root, descriptors)

    def path(self, name: str) -> Path:
        try:
            path = self.root / self.descriptors[name]["path"]
        except KeyError as exc:
            raise KeyError(f"Artifact {name!r} is unavailable") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def frame(self, name: str) -> pd.DataFrame:
        path = self.path(name)
        return _read_parquet(str(path), path.stat().st_mtime_ns)

    def json(self, name: str) -> dict:
        path = self.path(name)
        return _read_json(str(path), path.stat().st_mtime_ns)

    def available(self, name: str) -> bool:
        return name in self.descriptors and (self.root / self.descriptors[name]["path"]).is_file()
```

```python
# pitch_oracle_core/ui/context.py
from dataclasses import dataclass
from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.ui.repository import ArtifactRepository


@dataclass(frozen=True)
class AppContext:
    config: LeagueConfig
    repository: ArtifactRepository
    capabilities: dict[str, dict]
    edition_id: str

    def has_capability(self, name: str) -> bool:
        return self.capabilities.get(name, {}).get("status") in {
            "available", "degraded"
        }
```

This also makes page tests straightforward: construct `AppContext` over synthetic
artifacts and call the renderer.

## 2. Capability-driven page registry

Every league gets the same navigation contract. Optional pages appear only when their
required artifacts/capabilities exist; individual optional cards degrade inside a
universal page.

```python
# pitch_oracle_core/ui/navigation.py
from dataclasses import dataclass
from collections.abc import Callable
import streamlit as st


@dataclass(frozen=True)
class PageSpec:
    group: str
    title: str
    icon: str
    path: str
    render: Callable
    required_artifacts: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()


def enabled(page: PageSpec, context) -> bool:
    return (
        all(context.repository.available(name) for name in page.required_artifacts)
        and all(context.has_capability(name) for name in page.required_capabilities)
    )


def build_navigation(context, page_specs: tuple[PageSpec, ...]):
    groups: dict[str, list] = {}
    for spec in page_specs:
        if not enabled(spec, context):
            continue

        def renderer(active=spec):
            active.render(context)

        groups.setdefault(spec.group, []).append(st.Page(
            renderer,
            title=spec.title,
            icon=spec.icon,
            url_path=spec.path,
            default=spec.path == "overview",
        ))
    return st.navigation(groups)
```

Do not make `Market Lab` visible with a page full of “N/A” when live odds are absent.
Do keep the Match Center visible if weather is absent; only its weather card changes.

## 3. Shared formatting

Probabilities remain numeric until the last rendering boundary.

```python
# pitch_oracle_core/ui/formatters.py
import math


def probability(value: float | None, digits: int = 0) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}%}"


def signed(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):+.{digits}f}"


def freshness(age_minutes: float) -> tuple[str, str]:
    if age_minutes <= 90:
        return "Fresh", "green"
    if age_minutes <= 360:
        return "Aging", "orange"
    return "Stale", "red"
```

The current `_display_predictions` converts probabilities into strings and then
parses the strings back to style them. New components should never do that.

## 4. Probability header with uncertainty (F01, F07, F46)

```python
# pitch_oracle_core/ui/components/probability.py
import streamlit as st
from pitch_oracle_core.ui.formatters import probability


def render_probability_header(forecast) -> None:
    outcomes = (
        ("Home win", forecast.p_home, forecast.p_home_lower80, forecast.p_home_upper80),
        ("Draw", forecast.p_draw, forecast.p_draw_lower80, forecast.p_draw_upper80),
        ("Away win", forecast.p_away, forecast.p_away_lower80, forecast.p_away_upper80),
    )
    columns = st.columns(3)
    for column, (label, point, lower, upper) in zip(columns, outcomes):
        column.metric(label, probability(point), help=(
            f"80% model interval: {probability(lower)}–{probability(upper)}"
        ))
        column.progress(float(point), text=(
            f"80% interval {probability(lower)}–{probability(upper)}"
        ))

    badges = []
    badges.append("🟢 Full history" if forecast.cold_start == "full" else "🟠 " + forecast.cold_start_label)
    badges.append("Stable" if forecast.leader_stability >= 0.75 else "Fragile")
    badges.append(f"Model {forecast.model_id}")
    st.caption(" · ".join(badges))
```

Risk can remain an ambiguity measure, but rename it **forecast ambiguity** and show it
next to empirical uncertainty. Do not imply it measures betting risk.

## 5. Scoreline heatmap and goal ladder (F02, F03)

```python
# pitch_oracle_core/ui/components/score_matrix.py
import numpy as np
import plotly.express as px
import streamlit as st


def render_score_matrix(matrix: np.ndarray, home: str, away: str, shown_goals: int = 6) -> None:
    values = np.asarray(matrix, dtype=float)[:shown_goals + 1, :shown_goals + 1]
    figure = px.imshow(
        values * 100.0,
        x=[str(goal) for goal in range(values.shape[1])],
        y=[str(goal) for goal in range(values.shape[0])],
        labels={"x": f"{away} goals", "y": f"{home} goals", "color": "Probability %"},
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="Blues",
    )
    figure.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_colorbar=dict(title="%"),
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    shown_mass = values.sum()
    if shown_mass < 0.999:
        st.caption(f"{1 - shown_mass:.2%} probability lies outside the displayed score grid.")


def goal_market_frame(markets: dict[str, float]):
    import pandas as pd
    rows = []
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        suffix = str(line).replace(".", "_")
        rows.append({
            "Line": line,
            "Over": markets[f"p_over_{suffix}"],
            "Under": markets[f"p_under_{suffix}"],
        })
    return pd.DataFrame(rows)
```

Render the goal ladder with `st.dataframe` progress columns or a grouped bar chart.
The values come from the matrix and require no separately fitted page logic.

## 6. Forecast drivers (F04, F05)

```python
# pitch_oracle_core/ui/components/drivers.py
import pandas as pd
import plotly.express as px
import streamlit as st


def render_drivers(drivers: pd.DataFrame, outcome: str) -> None:
    selected = (
        drivers.loc[drivers["outcome"] == outcome]
        .assign(abs_contribution=lambda frame: frame["contribution"].abs())
        .nlargest(8, "abs_contribution")
        .sort_values("contribution")
    )
    if selected.empty:
        st.info("Structured forecast drivers are not available for this model.")
        return
    figure = px.bar(
        selected,
        x="contribution",
        y="display_name",
        orientation="h",
        color="contribution",
        color_continuous_scale=["#b42318", "#f2f4f7", "#027a48"],
        labels={"contribution": "Probability contribution", "display_name": ""},
        hover_data=["value", "sample_timestamp", "source"],
    )
    figure.update_layout(margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def evidence_bullets(drivers: pd.DataFrame, caveat: str) -> list[str]:
    strongest = drivers.assign(abs_value=drivers.contribution.abs()).nlargest(2, "abs_value")
    bullets = [
        f"**{row.display_name}:** {row.explanation}"
        for row in strongest.itertuples()
    ]
    bullets.append(f"**Uncertainty:** {caveat}")
    return bullets
```

Explanations are generated during artifact build and contain `display_name`,
`definition`, `value`, `contribution`, `sample_timestamp`, `source`, and a templated
`explanation`. The UI does not invent causal language.

## 7. Fixture selection and stable links (F01, F11, F12)

```python
# pitch_oracle_core/ui/pages/match_center.py
import pandas as pd
import streamlit as st


def selected_fixture(fixtures: pd.DataFrame) -> str:
    requested = st.query_params.get("fixture")
    valid_ids = set(fixtures.fixture_id.astype(str))
    if requested in valid_ids:
        default_index = fixtures.fixture_id.astype(str).tolist().index(requested)
    else:
        default_index = 0
    labels = fixtures.apply(
        lambda row: f"{row.home_display_name} vs {row.away_display_name} · {row.kickoff_local}",
        axis=1,
    )
    label_to_id = dict(zip(labels, fixtures.fixture_id.astype(str)))
    choice = st.selectbox("Fixture", labels.tolist(), index=default_index)
    fixture_id = label_to_id[choice]
    st.query_params["fixture"] = fixture_id
    return fixture_id


def render(context) -> None:
    fixtures = context.repository.frame("fixtures")
    forecasts = context.repository.frame("forecasts")
    fixture_id = selected_fixture(fixtures.loc[fixtures.status == "scheduled"])
    fixture = fixtures.set_index("fixture_id").loc[fixture_id]
    forecast = forecasts.set_index("fixture_id").loc[fixture_id]
    # Typed adapters convert Series into FixtureView / ForecastView before render.
    render_match_header(fixture, forecast)
    render_probability_header(forecast)
    left, right = st.columns([1.25, 1])
    with left:
        render_score_matrix(load_score_matrix(context, fixture_id), fixture.home_display_name, fixture.away_display_name)
    with right:
        render_goal_outlook(context, fixture_id)
    render_evidence(context, fixture_id)
    render_team_form_comparison(context, fixture_id)
    render_season_stakes(context, fixture_id)
    render_optional_context(context, fixture_id)
```

The helper functions above map directly to component modules. Keep page renderers
short enough to understand in one screen.

## 8. Team trend chart (F13–F16, F21)

```python
# pitch_oracle_core/ui/components/team_trends.py
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_team_trend(events: pd.DataFrame, team_name: str, metric: str, label: str) -> None:
    frame = events.sort_values("kickoff_utc").copy()
    frame["rolling"] = frame[metric].ewm(halflife=5, min_periods=1).mean()
    figure = go.Figure()
    figure.add_scatter(
        x=frame.kickoff_utc, y=frame[metric], mode="markers",
        marker_color=frame.result.map({"W": "#027a48", "D": "#667085", "L": "#b42318"}),
        name="Match",
        customdata=frame[["opponent_name", "venue_role", "score"]],
        hovertemplate="%{x|%b %d}: %{y:.2f}<br>%{customdata[1]} vs %{customdata[0]} · %{customdata[2]}<extra></extra>",
    )
    figure.add_scatter(
        x=frame.kickoff_utc, y=frame.rolling, mode="lines", name="EW trend",
        line=dict(color="#1554a6", width=3),
    )
    figure.update_layout(
        title=f"{team_name} · {label}", yaxis_title=label,
        margin=dict(l=10, r=10, t=45, b=10), legend_orientation="h",
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
```

Controls should offer last 5/10/20/all, overall/home/away, raw/opponent-adjusted, and
metric selection. Always show `n` and preserve the same y-axis when comparing teams.

## 9. Season projection table and race chart (F27–F31)

```python
# pitch_oracle_core/ui/components/projection_table.py
import pandas as pd
import plotly.express as px
import streamlit as st


def render_projection_table(projections: pd.DataFrame, outcome_columns: list[str]) -> None:
    display = projections[[
        "current_position", "team_name", "current_points", "expected_points",
        "expected_position", *outcome_columns,
    ]].sort_values("expected_position")
    column_config = {
        "current_position": st.column_config.NumberColumn("Now", format="%d"),
        "team_name": st.column_config.TextColumn("Team", pinned=True),
        "current_points": st.column_config.NumberColumn("Pts", format="%d"),
        "expected_points": st.column_config.NumberColumn("Expected pts", format="%.1f"),
        "expected_position": st.column_config.NumberColumn("Expected rank", format="%.1f"),
    }
    column_config.update({
        column: st.column_config.ProgressColumn(
            column.replace("p_", "").replace("_", " ").title(), min_value=0.0,
            max_value=1.0, format="percent",
        ) for column in outcome_columns
    })
    st.dataframe(display, hide_index=True, width="stretch", column_config=column_config)


def render_position_distribution(position_probability: pd.DataFrame) -> None:
    figure = px.imshow(
        position_probability.set_index("team_name") * 100,
        labels={"x": "Final position", "y": "", "color": "Probability %"},
        aspect="auto", color_continuous_scale="Blues",
    )
    figure.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
```

Outcome columns come from `CompetitionRules.outcome_labels`, so Belgium can render
Champions' Play-off qualification while another edition renders top four or safety.

## 10. Model Lab (F36–F45)

Model Lab should have four tabs:

1. **Deployment** — champion, challengers, release decision, data window, feature and
   rule versions.
2. **Performance** — rolling log loss/Brier, accuracy only as secondary context,
   baseline deltas and bootstrap intervals.
3. **Calibration** — reliability curves for H/D/A, confidence histogram, ECE, and
   interval coverage.
4. **Cohorts & drift** — promoted teams, early season, phase, rest, favorite bands,
   provider quality, feature/prediction drift.

```python
# pitch_oracle_core/ui/components/reliability.py
import plotly.graph_objects as go
import streamlit as st


def render_reliability(curves) -> None:
    figure = go.Figure()
    figure.add_scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
        line=dict(color="#98a2b3", dash="dash"),
    )
    colors = {"home": "#1554a6", "draw": "#667085", "away": "#b54708"}
    for outcome, frame in curves.groupby("outcome"):
        figure.add_scatter(
            x=frame.mean_forecast, y=frame.observed_rate,
            mode="lines+markers", name=outcome.title(),
            line=dict(color=colors[outcome]),
            customdata=frame[["n"]],
            hovertemplate="Forecast %{x:.0%}<br>Observed %{y:.0%}<br>n=%{customdata[0]}<extra></extra>",
        )
    figure.update_layout(
        xaxis_title="Mean forecast", yaxis_title="Observed frequency",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
```

Belgium's 0.31% full-evaluation draw recall should be impossible to miss on this page.

## 11. Data Control Room (F46–F48)

Replace the raw filename table with:

- overall publish status and generated time;
- active-team entity coverage with unresolved alias list;
- source capability cards with observed time, coverage, status, and error code;
- fixture/result/feature/model row and time coverage;
- quality checks grouped by blocking/warning/informational;
- artifact dependency graph and download links;
- “why is this card unavailable?” messages users can understand.

```python
def render_quality_checks(report: dict) -> None:
    import pandas as pd
    import streamlit as st

    checks = pd.DataFrame(report["checks"])
    blocking = checks.loc[checks.severity == "blocking"]
    if not blocking.empty:
        st.error(f"{len(blocking)} blocking data checks failed")
    else:
        st.success("All publication-blocking data checks passed")
    st.dataframe(
        checks[["status", "severity", "check", "observed", "expected", "message"]],
        hide_index=True, width="stretch",
        column_config={
            "status": st.column_config.TextColumn("Status", width="small"),
            "severity": st.column_config.TextColumn("Severity", width="small"),
            "check": st.column_config.TextColumn("Check", pinned=True),
            "message": st.column_config.TextColumn("Details", width="large"),
        },
    )
```

## 12. Scenario Lab safety (F06)

Only features declared scenario-mutable can be changed. Display the base and scenario
forecast side by side, the exact inputs changed, and the caveat that this is a model
sensitivity—not a new observed forecast.

```python
@dataclass(frozen=True)
class ScenarioControl:
    feature: str
    label: str
    minimum: float
    maximum: float
    step: float


def apply_scenario(base_features, changes, allowed_controls):
    allowed = {control.feature for control in allowed_controls}
    forbidden = set(changes).difference(allowed)
    if forbidden:
        raise ValueError(f"Scenario cannot mutate: {sorted(forbidden)}")
    scenario = base_features.copy()
    for feature, value in changes.items():
        scenario[feature] = value
    return scenario
```

Lineup, weather, and market controls do not appear when their provider capability or
model feature is absent.

## 13. Performance budget

The app must stay cache-first:

| Interaction | Budget | Technique |
|---|---:|---|
| Initial overview render | < 1.5 s warm / < 3 s cold | compact overview JSON/Parquet, lazy page imports |
| Fixture switch | < 300 ms warm | indexed frames, matrix lookup by fixture ID |
| Team/filter change | < 500 ms | cached Parquet reads and vectorized filters |
| Scenario recompute | < 750 ms | one-row fitted inference; no refitting |
| Projection page | < 500 ms | simulations precomputed in workflow |

Do not load Torch, SHAP, XGBoost, or fitted estimators merely to render cached pages.
Only Scenario Lab may load a lightweight production inference artifact on demand.

## 14. Accessibility and presentation rules

- Never encode probability/risk only by red/green; include labels and numbers.
- Maintain readable contrast in both existing day/night themes.
- Give every chart a table/download alternative.
- Use real team/competition display names while joining by IDs.
- Keep percentages consistent to one decimal in details and whole percent in headers.
- Do not use a progress bar to imply calibration or certainty it does not measure.
- Put generated/source timestamps beside the relevant block, not only in a footer.
- Avoid emojis as the sole semantic signal; icons supplement text.
- On narrow screens, use one-column flow and avoid dataframe-first match pages.

## 15. UI tests

The existing unit tests for table transforms should remain. Add:

- artifact repository v2 compatibility and v3 typed reads;
- navigation inclusion/exclusion by artifacts and capabilities;
- stable fixture query links;
- score matrix tail and market labels;
- cold-start and stale-data badges;
- rule-specific outcome column rendering from configuration;
- snapshot tests for deterministic evidence text;
- Streamlit AppTest flows for fixture/team selection and empty optional states;
- contrast and keyboard checks for custom HTML components;
- a consumer smoke test using Belgium aliases, rules, and artifacts.

