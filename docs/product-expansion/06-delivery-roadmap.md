# Delivery roadmap and code map

This is the order to implement the 50-feature catalog without destabilizing existing
consumers. It deliberately fixes Belgium's identity, time, and rules defects before
changing the visual experience.

## Work package 0 — Baselines and contract freeze

**Goal:** make current behavior reproducible before changing schemas.

1. Record current core tests, wheel metadata, consumer contract, and artifact hashes.
2. Add a Belgium fixture to core tests containing the 18 active ESPN names and their
   expected canonical IDs.
3. Save the current Belgium out-of-fold evaluation predictions, not only aggregate
   metrics.
4. Add the package-version consistency test and choose `_version.py` as the single
   version source (or use a VCS-based version tool).
5. Record page startup and fixture-filter performance budgets.

Exit gates:

- the branch tests pass on Python 3.12 and 3.13;
- current Belgium artifacts validate under manifest v2;
- a frozen audit report documents current log loss, Brier, calibration, draw recall,
  entity coverage, unique forecast vectors, and page inventory.

## Work package 1 — Canonical data foundation (P0)

**Implements:** F26, F37, F43, F46–F48 and prerequisites for everything else.

Add:

```text
pitch_oracle_core/domain/entities.py
pitch_oracle_core/domain/competitions.py
pitch_oracle_core/data/normalization.py
pitch_oracle_core/data/validation.py
pitch_oracle_core/features/ledger.py
pitch_oracle_core/features/registry.py
pitch_oracle_core/competition/standings.py
pitch_oracle_core/artifacts/manifest.py
pitch_oracle_core/pipelines/build_consumer.py
```

Tasks:

1. Build canonical team and alias tables for all configured consumers.
2. Change fixture/result normalization to emit IDs, UTC kickoff, edition ID, source,
   and observed time.
3. Replace calendar-year seasons and role-split team state with the team ledger.
4. Implement edition-versioned rule adapters and verify current standings.
5. Generate manifest v3 alongside v2 during one compatibility release.
6. Add structured provider-run and quality-report artifacts.
7. Block publication on unresolved active teams, timestamp leakage, invalid rules,
   and incoherent probabilities.

Belgium exit gates:

- 18/18 current fixture teams resolve;
- established teams no longer receive league priors because of display-name mismatch;
- exact aliases survive accent and punctuation differences without fuzzy auto-accept;
- live standings respect the configured competition-edition rules;
- kickoff artifacts retain UTC and render correctly in Brussels and user timezones;
- every forecast exposes history counts and prior weight.

## Work package 2 — Match Intelligence release (P1)

**Implements:** F01–F05, F07–F10, F12, F38.

Add:

```text
pitch_oracle_core/domain/forecasts.py
pitch_oracle_core/ui/context.py
pitch_oracle_core/ui/repository.py
pitch_oracle_core/ui/navigation.py
pitch_oracle_core/ui/components/{probability,score_matrix,drivers,freshness}.py
pitch_oracle_core/ui/pages/{overview,match_center,radars,prediction_history}.py
```

Tasks:

1. Introduce the joint score-matrix forecast contract.
2. Backfill current Poisson output into that contract for immediate parity.
3. Generate market ladders and deterministic evidence artifacts.
4. Build stable fixture links, uncertainty/cold-start badges, and freshness panels.
5. Replace the current overview with matchday pulse, radar lists, and race snapshot.
6. Add append-only forecast issuance history.

Exit gates:

- every fixture page renders without loading training libraries;
- displayed 1X2 and market probabilities reconcile to the matrix;
- all explanation claims identify a source metric and pre-kickoff timestamp;
- a fixture with only priors is clearly marked;
- app performance meets the budgets in `05-ui-implementation.md`.

## Work package 3 — Team and league command centers (P1/P2)

**Implements:** F13–F22, F26–F35.

Add:

```text
pitch_oracle_core/analytics/{team_snapshots,league_trends,storylines}.py
pitch_oracle_core/competition/{simulation,stakes}.py
pitch_oracle_core/ui/components/{team_trends,projection_table}.py
pitch_oracle_core/ui/pages/{team_center,comparison,standings,projections,league_lab}.py
```

Tasks:

1. Produce team snapshot and event marts from the common ledger.
2. Fit/update dynamic Elo and expose historical rating states.
3. Implement team center, comparison, adjusted form, rankings, and schedule difficulty.
4. Implement regular-season simulation, then generic split/playoff adapters.
5. Derive qualification/relegation labels and points thresholds from edition rules.
6. Generate matchday stakes and deterministic storyline artifacts.
7. Add cross-league export schema; build a separate index only after all consumers
   publish compatible artifacts.

Exit gates:

- team views count home and away matches in one chronological state;
- simulation is deterministic at a fixed seed and passes probability-sum tests;
- Belgium points transitions and Scotland pool splits pass scenario fixtures;
- no page checks `config.key` to choose a rule or outcome label.

## Work package 4 — Forecast Foundry (P1/P2)

**Implements:** F36–F45.

Add:

```text
pitch_oracle_core/models/{elo,dixon_coles,stacking,uncertainty}.py
pitch_oracle_core/evaluation/{backtest,calibration,cohorts,drift,registry}.py
pitch_oracle_core/ui/components/reliability.py
pitch_oracle_core/ui/pages/model_lab.py
```

Tasks:

1. Make the existing Poisson and class-prior models registered baselines.
2. Add dynamic Elo and time-decayed Dixon-Coles as challengers.
3. Persist rolling-origin predictions for every candidate.
4. Add coherent calibration, bootstrap intervals, cohort metrics, and drift.
5. Add out-of-fold stacking; promote only if it passes the contextual gate.
6. Render structured model registry/evaluation artifacts in Model Lab.
7. Stop training neural/LSTM/XGBoost artifacts by default when they are neither a
   registered challenger nor used in production. An opt-in experiment profile may
   continue to train them.
8. Implement the common `ScoreModel`/`ProbabilityGrid` protocol and proper-score panel
   from `10-research-implementation.md` before adding further candidate families.
9. Freeze distribution-tournament candidates and run paired chronological comparisons;
   when many specifications are searched, record a White-style/SPA family test and an
   untouched forward result.
10. Add bivariate, diagonal-inflated, NB, CMP, state-space, or hierarchical candidates
    only in the order and under the diagnostic gates in R01–R10.

Exit gates:

- a reproduction command recreates every metric from persisted evaluation rows;
- model selection is deterministic and records why the champion won;
- draw performance and calibration are explicit release criteria;
- no model is promoted on accuracy alone;
- no model is promoted on simulated ROI alone;
- log loss, Brier, RPS, calibration/sharpness, paired uncertainty, and operational
  failure rates are visible in one decision record;
- inference never loads an unused neural artifact.

## Work package 5 — Optional context and Market Lab (P2/P3)

**Implements:** F06, F11, F20, F22–F25, F49–F50.

Tasks:

1. Introduce timestamped snapshots for weather, squad availability, manager tenure,
   referee assignment, venue, travel, and odds.
2. Build scenario controls only for features the production model consumes.
3. Add fixture revision timeline and before/after forecast deltas.
4. Add style clusters with a published metric dictionary and stability report.
5. Add market de-vigging, consensus/dispersion, fair prices, and movement.
6. Add portfolio backtesting and constrained fractional Kelly only after executable,
   timestamped price coverage is reliable.
7. Maintain distinct `independent` and `market_aware` registry tracks, compare both
   with a de-vigged market baseline, and use exact quarter-line settlement.
8. Apply family-level data-snooping controls when selecting among markets, bookmakers,
   edge thresholds, or staking policies.

Exit gates:

- stale optional data never silently enters a fresh forecast;
- all optional cards identify provider, observed time, and coverage;
- Market Lab disappears when odds capability is unavailable;
- portfolio defaults produce zero stake when freshness, uncertainty, or edge gates fail.

## Feature-to-code ownership map

| Features | Primary modules/artifacts |
|---|---|
| F01, F05, F12 | `ui/pages/match_center.py`, `forecast_explanations.parquet` |
| F02, F03, F43 | `domain/forecasts.py`, `ui/components/score_matrix.py`, `score_matrices` |
| F04 | `evaluation/explanations.py`, `ui/components/drivers.py` |
| F06 | `features/registry.py`, `ui/pages/scenario_lab.py`, production inference adapter |
| F07 | `models/uncertainty.py`, forecast interval columns |
| F08–F10 | `analytics/radars.py`, `ui/pages/radars.py` |
| F11, F38 | append-only `forecast_ledger.parquet`, prediction-history page |
| F13–F16 | `features/ledger.py`, `analytics/team_snapshots.py`, team-center page |
| F17–F18 | comparison/H2H projections over `team_events.parquet` |
| F19, F40 | `models/elo.py`, `rating_history.parquet` |
| F20 | `analytics/style_clusters.py`, cluster stability artifact |
| F21–F22 | `analytics/schedule.py`, fixtures, venue coordinates |
| F23–F25 | optional manager/squad/referee snapshots and cards |
| F26 | `competition/standings.py`, `CompetitionRules` |
| F27–F31 | `competition/simulation.py`, `season_simulations.parquet` |
| F32–F35 | league/storyline/cross-league analytical marts |
| F36 | `evaluation/registry.py`, `model_registry.json` |
| F37 | evaluation predictions, calibration artifact/component |
| F39 | `models/dixon_coles.py`, parameter artifact |
| F41 | `models/stacking.py`, OOF component predictions |
| F42, F46 | entity history, promoted-team priors, cold-start metadata |
| F44–F45 | drift/cohort evaluators and Model Lab tabs |
| F47–F48 | provider runs, quality report, manifest v3, Data Control Room |
| F49–F50 | odds snapshots, market and portfolio services/pages |

## Backward-compatible artifact migration

Release in three steps:

### Release A — dual read, v2 write

- add the new repository abstraction;
- continue reading current CSV/pickle artifacts;
- add package-version consistency and entity-coverage warnings;
- no consumer migration required.

### Release B — v3 write, dual read

- workflow writes canonical v3 artifacts and a v3 manifest;
- app can read v2 or v3, but new pages require v3 descriptors;
- consumer verification validates both during the transition;
- issue warnings with a removal version for v2.

### Release C — v3 required

- consumer template and reusable workflow require v3;
- remove inference based on “latest row whose feature name starts with Home/Away”;
- remove legacy page reads and obsolete artifacts after all consumers pin the release.

Do not overwrite a valid v2 artifact set in place. Write v3 to new paths, validate the
complete graph, then atomically replace the manifest pointer.

## Dependency changes

Core can keep lightweight base dependencies. Suggested extras:

```toml
[project.optional-dependencies]
data = ["pyarrow>=18"]
ml = [
  "scikit-learn>=1.7,<2",
  "scipy>=1.11",
  "xgboost>=2"
]
app = [
  "streamlit>=1.57",
  "plotly>=5",
  "pyarrow>=18"
]
neural = ["torch>=2"]
```

Make `neural` an experiment dependency, not part of every consumer runtime. The
current consumer extra installs Torch, SHAP, seaborn, and scraping libraries even
though the cache-first app does not need them to render. Split extras into:

- `runtime`: Streamlit, Plotly, PyArrow;
- `pipeline`: scikit-learn, SciPy, XGBoost, providers;
- `diagnostics`: SHAP and plotting;
- `experiments-neural`: Torch.

## Tests to implement first

```python
# tests/test_entity_coverage_belgium.py
from datetime import date


def test_all_current_belgium_espn_names_resolve(belgium_resolver):
    names = {
        "Anderlecht", "Antwerp", "Cercle Brugge KSV", "Club Brugge",
        "KAA Gent", "KV Kortrijk", "KV Mechelen", "KVC Westerlo",
        "Lommel SK", "OH Leuven", "RAAL La Louvière", "Racing Genk",
        "Royal Charleroi SC", "Sint-Truidense", "Standard Liege",
        "Union St.-Gilloise", "Waasland-Beveren", "Zulte-Waregem",
    }
    resolved = [
        belgium_resolver.resolve("espn", name, date(2026, 8, 10))
        for name in names
    ]
    assert all(item.team_id for item in resolved)
    assert len({item.team_id for item in resolved}) == 18
```

```python
# tests/test_team_event_state.py
import pandas as pd
from pitch_oracle_core.features.ledger import build_team_events, add_prior_team_state


def test_away_match_updates_next_home_rest_and_form():
    matches = pd.DataFrame([
        {"fixture_id": "f1", "edition_id": "x:2026-27", "kickoff_utc": "2026-08-01T14:00Z",
         "home_team_id": "a", "away_team_id": "b", "home_goals": 0, "away_goals": 2},
        {"fixture_id": "f2", "edition_id": "x:2026-27", "kickoff_utc": "2026-08-08T14:00Z",
         "home_team_id": "b", "away_team_id": "c", "home_goals": 1, "away_goals": 1},
    ])
    events = add_prior_team_state(build_team_events(matches))
    b_next = events.loc[(events.fixture_id == "f2") & (events.team_id == "b")].iloc[0]
    assert b_next.rest_days == 7
    assert b_next.points_l5 == 3
```

```python
# tests/test_season_identity.py
from datetime import datetime, timezone


def test_winter_match_remains_in_august_start_edition(belgium_edition):
    assert belgium_edition.season_id(
        datetime(2027, 2, 1, tzinfo=timezone.utc)
    ) == "2026-27"
```

```python
# tests/test_forecast_coherence.py
import numpy as np


def test_markets_derive_from_one_score_distribution(forecast):
    home, draw, away = forecast.one_x_two
    assert np.isclose(home + draw + away, 1.0)
    assert np.isclose(forecast.total_over(2.5) + (1 - forecast.total_over(2.5)), 1.0)
    assert 0 <= forecast.btts <= 1
```

Additional mandatory suites:

- property tests for probability, table, and simulation invariants;
- golden standings for each configured edition/rule adapter;
- time-travel tests proving no future feature/provider observation is visible;
- promoted-team/cold-start fixtures;
- rolling-origin reproducibility and paired model comparisons;
- manifest dependency, schema, hash, freshness, and atomic-publication tests;
- UI capability and empty-state tests;
- wheel/runtime/tag/version equality;
- Belgium consumer end-to-end artifact and AppTest smoke test.

## Reusable workflow target

After compatibility migration, the consumer workflow can shrink to:

```yaml
- name: Build consumer intelligence artifacts
  run: >-
    python -m pitch_oracle_core.pipelines.build_consumer
    --league "${PITCH_ORACLE_LEAGUE}"
    --root .
    --manifest-version 3

- name: Verify publication bundle
  run: python -m pitch_oracle_core.artifacts.verify --root . --strict

- name: Run consumer smoke tests
  run: python -m pytest -q
```

The pipeline internally records stage reports; workflow YAML should not duplicate the
pipeline graph.

## Rollout order across consumers

1. **Belgium first** for entity resolution and rule-aware standings because it
   currently demonstrates both defects.
2. **Netherlands second** to test a structurally simple competition and existing
   stadium/alias coverage.
3. **Scotland third** to validate split pools and cross-division playoff hooks.
4. **Turkey fourth** to validate accents, sanctions, and promoted-team priors.
5. **EPL fifth** to validate richer optional providers without making them baseline
   requirements.
6. **Portugal** when its fixture provider/config is consumer-ready.

The core release is the product release. Consumer changes should normally be a pin
bump plus generated entity/rule/artifact data.

## Feature flags and rollback

Use artifact-backed capability flags, not permanent environment toggles. During
rollout, temporary flags may select `legacy_pages` or `artifact_v3_pages`, but each
flag must have an owner and removal version.

Rollback rules:

- never downgrade/overwrite a valid artifact graph in place;
- retain the prior manifest and artifact directory for one release;
- app factory may select the last fully valid manifest when the newest build fails;
- never fall back across leagues or competition editions;
- show the serving artifact's generated time and version after rollback.

## Definition of done for the program

- all 50 catalog features have a shipped, intentionally deferred, or capability-
  unavailable status in `model_registry.json`/manifest capabilities;
- every P0 and P1 feature is live in Belgium and one simple-format league;
- current fixture entity coverage is 100% in all consumers;
- no league-specific page/model branch is needed for rules already expressible by
  `CompetitionRules`;
- forecasts, goal markets, and simulations share one coherent score distribution;
- Model Lab reproduces production selection from out-of-time rows;
- Data Control Room exposes lineage, freshness, coverage, and blocking checks;
- runtime dependencies and startup cost exclude unused training frameworks;
- `app_shell.py` and v2-only reads are removed after migration parity;
- consumer pin upgrades require no copied analysis code.

## Immediate first pull requests

Keep the first PRs reviewable:

1. **Canonical Belgium identities + coverage gate.** Add entity contracts, alias data,
   resolver tests, and prediction cold-start metadata; do not change models yet.
2. **UTC fixtures + edition identity.** Replace `US/Eastern`, preserve UTC, add local
   render selection and season-ID tests.
3. **Team-event ledger.** Implement and compare old/new features, then switch form and
   rest behind a feature-policy version bump.
4. **Rule-aware standings.** Edition rules, Belgium golden table, Scotland golden
   split fixture, and new standings page.
5. **Forecast v3 + Match Center.** Joint score contract, matrix/market invariants,
   artifact repository, and first new page.

Each PR should include ordinary Markdown bullet points in its description, per the
repository instructions.
