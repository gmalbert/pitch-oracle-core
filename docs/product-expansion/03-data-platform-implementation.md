# Data and platform implementation

This document contains the shared contracts and core algorithms that unlock F01–F50.
The snippets are intended as the starting implementation, not pseudocode. They use
the existing Python stack (`dataclasses`, pandas, NumPy) and fit the current
cache-first consumer architecture.

## Proposed package layout

```text
pitch_oracle_core/
  domain/
    entities.py             # canonical IDs and aliases
    competitions.py         # editions, phases, tie-breakers, outcome labels
    forecasts.py            # score distributions and forecast contracts
  data/
    providers.py            # provider run/capability contracts
    normalization.py        # canonical column and identity mapping
    validation.py           # quality gates
  features/
    ledger.py               # perspective-normalized team events
    registry.py             # point-in-time feature definitions
    builders.py             # match/team/league feature marts
  artifacts/
    manifest.py             # manifest v3 and typed datasets
    repository.py           # app-side readers
  competition/
    standings.py            # rule-aware tables
  pipelines/
    build_consumer.py       # explicit staged orchestration
```

Keep compatibility modules (`prepare_model_data.py`, `precompute_database.py`) as
thin command wrappers until every consumer has upgraded.

## 1. Canonical entities and provider aliases

Display names must never be keys. The canonical registry is small enough to remain a
reviewable CSV/Parquet artifact in each consumer at first; it can move to a service
later without changing the contract.

```python
# pitch_oracle_core/domain/entities.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
import re
import unicodedata


class ResolutionStatus(StrEnum):
    EXACT_CANONICAL = "exact_canonical"
    PROVIDER_ALIAS = "provider_alias"
    NEW_TEAM_PRIOR = "new_team_prior"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Team:
    team_id: str                 # e.g. "bel:club-brugge"
    canonical_name: str
    country_code: str
    founded: int | None = None


@dataclass(frozen=True)
class TeamAlias:
    provider: str                # "espn", "football_data_uk", "clubelo"
    external_name: str
    team_id: str
    valid_from: date | None = None
    valid_to: date | None = None

    def valid_on(self, when: date) -> bool:
        return (
            (self.valid_from is None or self.valid_from <= when)
            and (self.valid_to is None or when <= self.valid_to)
        )


@dataclass(frozen=True)
class Resolution:
    raw_name: str
    team_id: str | None
    status: ResolutionStatus
    provider: str


def normalized_name(value: str) -> str:
    """Normalize only for lookup; never overwrite the display name."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


class EntityResolver:
    def __init__(self, teams: list[Team], aliases: list[TeamAlias]) -> None:
        self._teams = {team.team_id: team for team in teams}
        self._canonical = {
            normalized_name(team.canonical_name): team.team_id for team in teams
        }
        self._aliases: dict[tuple[str, str], list[TeamAlias]] = {}
        for alias in aliases:
            if alias.team_id not in self._teams:
                raise ValueError(f"Alias references unknown team {alias.team_id!r}")
            key = (alias.provider, normalized_name(alias.external_name))
            self._aliases.setdefault(key, []).append(alias)

    def resolve(self, provider: str, raw_name: str, when: date) -> Resolution:
        candidates = [
            alias for alias in self._aliases.get(
                (provider, normalized_name(raw_name)), []
            )
            if alias.valid_on(when)
        ]
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous aliases for {provider}:{raw_name!r} on {when}"
            )
        if candidates:
            return Resolution(
                raw_name, candidates[0].team_id,
                ResolutionStatus.PROVIDER_ALIAS, provider,
            )
        team_id = self._canonical.get(normalized_name(raw_name))
        if team_id:
            return Resolution(
                raw_name, team_id, ResolutionStatus.EXACT_CANONICAL, provider
            )
        return Resolution(raw_name, None, ResolutionStatus.UNRESOLVED, provider)
```

Belgium's initial alias patch becomes data, not Python conditionals:

```csv
provider,external_name,team_id,valid_from,valid_to
espn,Cercle Brugge KSV,bel:cercle-brugge,,
espn,KAA Gent,bel:gent,,
espn,KV Kortrijk,bel:kortrijk,,
espn,KV Mechelen,bel:mechelen,,
espn,KVC Westerlo,bel:westerlo,,
espn,OH Leuven,bel:oh-leuven,,
espn,RAAL La Louvière,bel:raal-la-louviere,,
espn,Racing Genk,bel:genk,,
espn,Royal Charleroi SC,bel:charleroi,,
espn,Sint-Truidense,bel:st-truiden,,
espn,Standard Liege,bel:standard-liege,,
espn,Union St.-Gilloise,bel:union-sg,,
espn,Waasland-Beveren,bel:beveren,,
espn,Zulte-Waregem,bel:waregem,,
```

CI may suggest close names for review, but it must not silently accept fuzzy matches.
The coverage gate is deterministic:

```python
def assert_active_team_coverage(resolutions: list[Resolution]) -> None:
    unresolved = sorted({item.raw_name for item in resolutions if item.team_id is None})
    if unresolved:
        raise RuntimeError(
            "Unresolved active team aliases: " + ", ".join(unresolved)
        )
```

## 2. Competition editions, seasons, and kickoff time

Use UTC for storage and edition-local time only for display/rules.

```python
# pitch_oracle_core/domain/competitions.py
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class CompetitionEdition:
    edition_id: str              # "bel.1:2026-27"
    competition_id: str          # "bel.1"
    display_name: str
    timezone: str
    season_start_month: int
    team_ids: tuple[str, ...]
    rules_version: str

    def season_id(self, kickoff_utc: datetime) -> str:
        local = kickoff_utc.astimezone(ZoneInfo(self.timezone))
        start = local.year if local.month >= self.season_start_month else local.year - 1
        return f"{start}-{str(start + 1)[-2:]}"


def parse_provider_kickoff(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Provider kickoff must include an offset")
    return parsed.astimezone(ZoneInfo("UTC"))


def kickoff_for_display(kickoff_utc: datetime, timezone: str) -> datetime:
    return kickoff_utc.astimezone(ZoneInfo(timezone))
```

Change fixture artifacts from `Date` + `Time` to this minimum contract:

```text
fixture_id, edition_id, kickoff_utc, home_team_id, away_team_id,
home_display_name, away_display_name, status, venue_id, provider_event_id,
source, observed_at
```

The app can offer `Competition local`, `My timezone`, and `UTC`; it must not bake
`US/Eastern` into ingestion.

## 3. Perspective-normalized team-event ledger

This replaces role-split form, rest, H2H, and season-state calculations. One match
becomes two rows, one from each team's perspective.

```python
# pitch_oracle_core/features/ledger.py
from __future__ import annotations

import numpy as np
import pandas as pd


MATCH_COLUMNS = {
    "fixture_id", "edition_id", "kickoff_utc", "home_team_id", "away_team_id",
    "home_goals", "away_goals",
}


def build_team_events(matches: pd.DataFrame) -> pd.DataFrame:
    missing = MATCH_COLUMNS.difference(matches.columns)
    if missing:
        raise ValueError(f"Missing match columns: {sorted(missing)}")

    frame = matches.copy()
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True)
    if frame["kickoff_utc"].isna().any():
        raise ValueError("Every match requires a valid kickoff_utc")

    home = pd.DataFrame({
        "fixture_id": frame["fixture_id"],
        "edition_id": frame["edition_id"],
        "kickoff_utc": frame["kickoff_utc"],
        "team_id": frame["home_team_id"],
        "opponent_id": frame["away_team_id"],
        "venue_role": "home",
        "goals_for": frame["home_goals"],
        "goals_against": frame["away_goals"],
    })
    away = pd.DataFrame({
        "fixture_id": frame["fixture_id"],
        "edition_id": frame["edition_id"],
        "kickoff_utc": frame["kickoff_utc"],
        "team_id": frame["away_team_id"],
        "opponent_id": frame["home_team_id"],
        "venue_role": "away",
        "goals_for": frame["away_goals"],
        "goals_against": frame["home_goals"],
    })
    events = pd.concat([home, away], ignore_index=True)
    events["goal_diff"] = events["goals_for"] - events["goals_against"]
    # Scheduled placeholder rows keep outcome fields null. This matters when
    # several future fixtures are built together: an unplayed game must not look
    # like a zero-point result in the next fixture's rolling state.
    events["points"] = np.select(
        [
            events["goal_diff"] > 0,
            events["goal_diff"] == 0,
            events["goal_diff"] < 0,
        ],
        [3.0, 1.0, 0.0],
        default=np.nan,
    )
    return events.sort_values(
        ["team_id", "kickoff_utc", "fixture_id"], kind="stable"
    ).reset_index(drop=True)


def _prior_rolling(
    frame: pd.DataFrame, value: str, window: int, aggregation: str
) -> pd.Series:
    grouped = frame.groupby("team_id", sort=False)[value]
    if aggregation == "sum":
        return grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=1).sum()
        )
    if aggregation == "mean":
        return grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=1).mean()
        )
    raise ValueError(f"Unsupported aggregation {aggregation!r}")


def add_prior_team_state(events: pd.DataFrame) -> pd.DataFrame:
    state = events.sort_values(
        ["team_id", "kickoff_utc", "fixture_id"], kind="stable"
    ).copy()
    state["history_n"] = state.groupby("team_id", sort=False)["goals_for"].transform(
        lambda series: series.shift(1).notna().cumsum()
    )
    state["rest_days"] = (
        state.groupby("team_id", sort=False)["kickoff_utc"]
        .diff().dt.total_seconds().div(86_400)
    )
    state["points_l5"] = _prior_rolling(state, "points", 5, "sum")
    state["goals_for_l5"] = _prior_rolling(state, "goals_for", 5, "mean")
    state["goals_against_l5"] = _prior_rolling(
        state, "goals_against", 5, "mean"
    )
    state["goal_diff_l10"] = _prior_rolling(state, "goal_diff", 10, "mean")
    completed_clean_sheet = np.where(
        state["goals_against"].notna(),
        (state["goals_against"] == 0).astype(float),
        np.nan,
    )
    state["clean_sheet"] = completed_clean_sheet
    state["clean_sheet_l10"] = _prior_rolling(
        state, "clean_sheet", 10, "mean",
    )
    return state


def match_feature_snapshots(events: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "fixture_id", "history_n", "rest_days", "points_l5", "goals_for_l5",
        "goals_against_l5", "goal_diff_l10", "clean_sheet_l10",
    ]
    home = events.loc[events["venue_role"] == "home", feature_columns].add_prefix("home_")
    away = events.loc[events["venue_role"] == "away", feature_columns].add_prefix("away_")
    return home.merge(
        away, left_on="home_fixture_id", right_on="away_fixture_id",
        validate="one_to_one",
    ).rename(columns={"home_fixture_id": "fixture_id"}).drop(
        columns="away_fixture_id"
    )
```

For a scheduled fixture, append two placeholder event rows with goals null, calculate
prior state, then select those rows. This guarantees training and inference use the
same builder. No “latest historical row whose column starts with Home” heuristic is
needed.

## 4. Point-in-time feature registry

The current centralized exclusion policy is good, but a denylist cannot prove
availability. Move to an allowlisted registry with lineage and availability time.

```python
# pitch_oracle_core/features/registry.py
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

import pandas as pd


class FeatureFamily(StrEnum):
    FORM = "form"
    STRENGTH = "strength"
    SCHEDULE = "schedule"
    SQUAD = "squad"
    WEATHER = "weather"
    MARKET = "market"


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    family: FeatureFamily
    dtype: str
    builder: Callable[[pd.DataFrame], pd.Series]
    max_age_hours: int | None = None
    optional_capability: str | None = None


class FeatureRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, FeatureDefinition] = {}

    def register(self, definition: FeatureDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Duplicate feature {definition.name!r}")
        self._definitions[definition.name] = definition

    def build(
        self,
        source: pd.DataFrame,
        *,
        kickoff_column: str = "kickoff_utc",
        observed_column: str = "observed_at",
    ) -> pd.DataFrame:
        kickoff = pd.to_datetime(source[kickoff_column], utc=True)
        observed = pd.to_datetime(source[observed_column], utc=True)
        leakage = observed >= kickoff
        if leakage.any():
            fixtures = source.loc[leakage, "fixture_id"].head(5).tolist()
            raise ValueError(f"Feature observations are not pre-kickoff: {fixtures}")
        return pd.DataFrame({
            name: definition.builder(source)
            for name, definition in self._definitions.items()
        }, index=source.index)

    def metadata(self) -> list[dict[str, object]]:
        return [
            {
                "name": item.name,
                "family": item.family.value,
                "dtype": item.dtype,
                "max_age_hours": item.max_age_hours,
                "optional_capability": item.optional_capability,
            }
            for item in self._definitions.values()
        ]
```

Persist `feature_timestamp`, `source_snapshot_id`, and `builder_version` alongside
training matrices. A feature definition change increments a feature-set version and
forces a rebuild.

## 5. Versioned competition rules and standings

Rules must vary by **edition**, because a league can change format without becoming a
new league key.

```python
from dataclasses import dataclass, field
from enum import StrEnum
from collections.abc import Iterable
import pandas as pd


class TieBreaker(StrEnum):
    POINTS = "points"
    GOAL_DIFFERENCE = "goal_difference"
    GOALS_FOR = "goals_for"
    HEAD_TO_HEAD_POINTS = "head_to_head_points"
    WINS = "wins"


@dataclass(frozen=True)
class PhaseRule:
    phase_id: str
    starts_after_round: int | None = None
    pool_sizes: tuple[int, ...] = ()
    pool_labels: tuple[str, ...] = ()
    points_multiplier: float = 1.0
    points_rounding: str = "none"


@dataclass(frozen=True)
class CompetitionRules:
    version: str
    win_points: int = 3
    draw_points: int = 1
    tie_breakers: tuple[TieBreaker, ...] = (
        TieBreaker.POINTS, TieBreaker.GOAL_DIFFERENCE, TieBreaker.GOALS_FOR,
    )
    phases: tuple[PhaseRule, ...] = (PhaseRule("regular"),)
    points_adjustments: dict[str, int] = field(default_factory=dict)
    outcome_labels: dict[str, tuple[int, ...]] = field(default_factory=dict)


def calculate_table(matches: pd.DataFrame, rules: CompetitionRules) -> pd.DataFrame:
    completed = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    events = build_team_events(completed)
    events["win"] = (events["goal_diff"] > 0).astype(int)
    events["draw"] = (events["goal_diff"] == 0).astype(int)
    events["loss"] = (events["goal_diff"] < 0).astype(int)
    events["points"] = (
        events["win"] * rules.win_points + events["draw"] * rules.draw_points
    )
    table = events.groupby("team_id", as_index=False).agg(
        played=("fixture_id", "count"), wins=("win", "sum"),
        draws=("draw", "sum"), losses=("loss", "sum"),
        goals_for=("goals_for", "sum"), goals_against=("goals_against", "sum"),
        points=("points", "sum"),
    )
    table["goal_difference"] = table["goals_for"] - table["goals_against"]
    table["points"] += table["team_id"].map(rules.points_adjustments).fillna(0)
    sort_columns = [item.value for item in rules.tie_breakers]
    unsupported = set(sort_columns).difference(table.columns)
    if unsupported:
        raise NotImplementedError(
            f"Tie-breakers require a specialized resolver: {sorted(unsupported)}"
        )
    table = table.sort_values(
        [*sort_columns, "team_id"],
        ascending=[False] * len(sort_columns) + [True], kind="stable",
    ).reset_index(drop=True)
    table.insert(0, "position", table.index + 1)
    return table
```

Head-to-head tie-breakers need a second pass over tied mini-leagues. Failing loudly
is preferable to silently presenting an incorrect generic table.

## 6. Provider capabilities and run health

Capability is more useful than a set of booleans because the UI needs coverage,
freshness, and failure reasons.

```python
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class CapabilityStatus(StrEnum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CapabilityReport:
    name: str
    status: CapabilityStatus
    source: str
    observed_at: datetime | None
    coverage: float
    message: str = ""

    @property
    def usable(self) -> bool:
        return self.status in {
            CapabilityStatus.AVAILABLE, CapabilityStatus.DEGRADED
        }


@dataclass(frozen=True)
class ProviderRun:
    run_id: str
    provider: str
    started_at: datetime
    finished_at: datetime
    rows_read: int
    rows_written: int
    status: str
    error_code: str | None = None
```

The page registry and individual cards consume `CapabilityReport`; they do not
inspect environment variables or catch broad provider exceptions.

## 7. Artifact manifest v3

Keep hashes, but add semantic metadata and dependencies.

```python
# pitch_oracle_core/artifacts/manifest.py
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import hashlib
import json


@dataclass(frozen=True)
class ArtifactDescriptor:
    name: str
    path: str
    media_type: str
    schema_name: str
    schema_version: int
    rows: int | None
    min_event_time: str | None
    max_event_time: str | None
    generated_at: str
    producer: str
    dependencies: tuple[str, ...] = ()
    model_id: str | None = None
    rules_version: str | None = None
    sha256: str = ""
    bytes: int = 0


@dataclass(frozen=True)
class ManifestV3:
    league: str
    edition_id: str
    core_version: str
    entity_registry_version: str
    generated_at: str
    artifacts: tuple[ArtifactDescriptor, ...]
    capabilities: tuple[dict[str, object], ...] = ()
    schema_version: int = 3


def file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest(), path.stat().st_size


def validate_dependency_graph(manifest: ManifestV3) -> None:
    names = {item.name for item in manifest.artifacts}
    if len(names) != len(manifest.artifacts):
        raise ValueError("Artifact names must be unique")
    for item in manifest.artifacts:
        missing = set(item.dependencies).difference(names)
        if missing:
            raise ValueError(f"{item.name} has missing dependencies: {missing}")


def write_manifest(manifest: ManifestV3, destination: Path) -> None:
    validate_dependency_graph(manifest)
    destination.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
```

Recommended v3 artifacts:

| Artifact | Format | Powers features |
|---|---|---|
| `entities/teams.parquet` + `aliases.parquet` | Parquet | F01, F13, F17, F46, F48 |
| `fixtures.parquet` | Parquet | All fixture/team/season pages |
| `team_events.parquet` | Parquet | F13–F25, F33 |
| `team_snapshots.parquet` | Parquet | F04, F13–F22 |
| `forecasts.parquet` | Parquet | F01, F07–F12, F38 |
| `score_matrices.npz` or long Parquet | NPZ/Parquet | F02, F03, F27–F31, F43 |
| `forecast_explanations.parquet` | Parquet | F04, F05 |
| `season_simulations.parquet` | Parquet | F27–F31 |
| `model_registry.json` | JSON | F36–F45 |
| `evaluation_predictions.parquet` | Parquet | F37, F38, F45 |
| `quality_report.json` | JSON | F46–F48 |
| `provider_runs.parquet` | Parquet | F47, F48 |
| `odds_snapshots.parquet` | Parquet | F49, F50 |

## 8. Explicit build pipeline

Each stage returns a report and writes atomically. The CLI composes functions; imports
do not run the pipeline.

```python
# pitch_oracle_core/pipelines/build_consumer.py
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
import os
import tempfile


@dataclass(frozen=True)
class StageReport:
    name: str
    rows: int
    output: Path
    warnings: tuple[str, ...] = ()


def atomic_output(destination: Path, writer: Callable[[Path], None]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent, suffix=destination.suffix
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_consumer(context) -> tuple[StageReport, ...]:
    reports = []
    reports.append(context.ingest_entities())
    reports.append(context.ingest_fixtures_and_results())
    reports.append(context.validate_canonical_data())
    reports.append(context.build_team_events())
    reports.append(context.build_feature_snapshots())
    reports.append(context.train_and_select_models())
    reports.append(context.build_forecasts_and_explanations())
    reports.append(context.run_season_simulations())
    reports.append(context.build_quality_and_drift_reports())
    reports.append(context.write_and_validate_manifest())
    return tuple(reports)
```

The GitHub reusable workflow becomes one `python -m pitch_oracle_core.pipelines ...`
call instead of a chain of import-time scripts. Stage reports remain available when a
later stage fails.

## 9. P0 validation code

Add these invariants before feature work:

```python
def validate_forecast_artifacts(
    fixtures: pd.DataFrame,
    forecasts: pd.DataFrame,
    scorelines: pd.DataFrame,
) -> None:
    scheduled = fixtures.loc[fixtures["status"] == "scheduled", "fixture_id"]
    if set(scheduled) != set(forecasts["fixture_id"]):
        raise ValueError("Scheduled fixture and forecast IDs disagree")

    probabilities = forecasts[["p_home", "p_draw", "p_away"]]
    if not np.isfinite(probabilities.to_numpy()).all():
        raise ValueError("Forecast probabilities must be finite")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("1X2 probabilities do not sum to one")

    mass = scorelines.groupby("fixture_id")["probability"].sum()
    if not np.allclose(mass.reindex(scheduled), 1.0, atol=1e-6):
        raise ValueError("Scoreline mass does not sum to one")

    derived = scorelines.assign(
        home=(scorelines.home_goals > scorelines.away_goals),
        draw=(scorelines.home_goals == scorelines.away_goals),
        away=(scorelines.home_goals < scorelines.away_goals),
    ).groupby("fixture_id").apply(
        lambda group: pd.Series({
            "p_home": group.loc[group.home, "probability"].sum(),
            "p_draw": group.loc[group.draw, "probability"].sum(),
            "p_away": group.loc[group.away, "probability"].sum(),
        }), include_groups=False,
    )
    joined = forecasts.set_index("fixture_id").join(
        derived, lsuffix="_forecast", rsuffix="_scores"
    )
    for outcome in ("home", "draw", "away"):
        if not np.allclose(
            joined[f"p_{outcome}_forecast"], joined[f"p_{outcome}_scores"],
            atol=1e-6,
        ):
            raise ValueError(f"Scoreline and 1X2 {outcome} probabilities disagree")
```

Also gate:

- canonical coverage for every active fixture team
- no duplicate `fixture_id` or provider event mapping
- `observed_at < kickoff_utc` for pre-match features
- no impossible negative goals, odds `<= 1`, or non-finite model inputs
- valid edition and rules version on every fixture
- freshness SLO by artifact type
- wheel/runtime/tag version agreement

These foundations are intentionally broader than one model or page. Once implemented,
most of the feature catalog becomes a set of small projections over trustworthy shared
state rather than another round of custom league scripts.
