# Repository audit: core and Belgium consumer

## Executive finding

The core is more capable than the current app looks. It already has cache integrity,
consumer bootstrapping, point-in-time safeguards, model ablation, probability
calibration, risk scoring, goal markets, phase configuration, weather support, and
provider protocols. The user-facing shell exposes only a thin slice of those assets.

The Belgium consumer is an especially useful test because it proves the package
boundary works while exposing the weakest part of the system: **canonical identity
and pre-match state assembly are not first-class platform concepts**. Fourteen of the
eighteen upcoming team names do not exactly match historical names. The production
Poisson model consequently falls back to league priors for many teams, and the UI
does not warn the user.

## What was reviewed

### Core runtime

| Area | Current implementation | Assessment |
|---|---|---|
| League contract | `pitch_oracle_core/config.py`, `leagues.py` | Good foundation; rules and providers need richer typed contracts. |
| Thin app factory | `app_factory.py`, `navigation.py` | Correct boundary; page registry should become capability-driven. |
| UI | `ui_pages.py` | Seven shared pages, mostly dataframes and headline metrics. |
| Feature policy | `features.py`, `prepare_model_data.py` | Several strong leakage guards; event ledger and season logic need redesign. |
| Training | `train_models.py`, `models/*` | Many fitted artifacts, but production selection is only `no_odds` vs Poisson. |
| Evaluation | `model_audit.py`, `poisson_evaluation.py` | Useful metrics and rolling-origin folds; missing context slices and uncertainty. |
| Prediction | `predictions.py` | Strict feature order and normalized output; cold-start is silent. |
| Markets | `goal_markets.py`, `best_bets.py`, `odds.py` | Useful primitives; no cross-market coherence or durable odds snapshots. |
| Competition phases | `phases.py` | Tested primitives exist; standings and simulation pages do not consume them. |
| Artifact contract | `cache.py`, reusable workflow | Strong integrity check; too few analytical artifacts and no per-artifact schema. |
| Optional sources | `providers.py`, `sources.py`, `weather.py` | Protocols exist; health/freshness/coverage are not modeled. |

### Belgium consumer

The repository is correctly thin:

```text
config.py                 # get_league_config("belgium")
predictions.py            # run_app(LEAGUE_CONFIG)
scripts/                  # bootstrap, precompute, verify
data_files/               # source and prediction artifacts
models/                   # fitted models and metadata
precomputed/              # feature, audit, SHAP, and manifest artifacts
tests/                     # consumer and artifact contracts
```

That is the right ownership model. New pages and models should be implemented in
core and enabled through artifacts/capabilities, not copied into Belgium.

## Quantitative Belgium snapshot

The reviewed artifacts were generated on 2026-08-11 with core `1.3.27`.

| Measure | Observed value | Meaning |
|---|---:|---|
| Historical matches | 1,250 | Five season labels, 2022-07-22 through 2026-08-09. |
| Engineered columns | 271 | 57 no-odds, 151 market, 63 excluded by the audit inventory. |
| Upcoming fixtures | 63 | 2026-08-14 through 2026-10-10. |
| Historical teams | 22 | Includes promoted/relegated clubs. |
| Upcoming teams | 18 | Current competition membership. |
| Exact identity matches | **4 / 18** | The dominant production-data defect. |
| Unique forecast vectors | 9 / 63 fixtures | Many fixtures share prior-driven rates. |
| Model feature width | 55 | Metadata says `feature_set: poisson` because Poisson won the release gate. |
| Poisson outcome accuracy | 48.08% | Useful, but accuracy alone is not enough. |
| Poisson log loss | 1.0396 | Better than the class-prior ablation. |
| Poisson multiclass Brier | 0.6240 | Better than the class-prior ablation. |
| Poisson draw recall | **0.31%** on full evaluation | The model almost never selects draws. |
| Rolling Poisson calibration error | 9.98% | Requires calibration and reliability views. |
| Model history rows | 2 | Both evaluations are identical and minutes apart. |

Examples of identity mismatches include:

| ESPN fixture name | Historical name |
|---|---|
| `Cercle Brugge KSV` | `Cercle Brugge` |
| `KAA Gent` | `Gent` |
| `KV Kortrijk` | `Kortrijk` |
| `Racing Genk` | `Genk` |
| `Union St.-Gilloise` | `St. Gilloise` |
| `Zulte-Waregem` | `Waregem` |
| `RAAL La Louvière` | `RAAL La Louviere` |

The current `belgium` config has no `team_aliases`, so `normalize_team_name` cannot
resolve those names before the Poisson dictionaries are queried.

## Current product surface

The package navigation provides:

1. Overview — three counts and an upcoming-fixtures table.
2. Predictions — filters, four counts, a colored table, and text expanders.
3. Standings — a basic W/D/L/GF/GA table.
4. Team Deep Dive — three cumulative metrics and ten recent matches in three table views.
5. Statistics — league outcome shares and top-ten goals scored.
6. Model Lab — latest Poisson accuracy/MAE and up to twelve history rows.
7. Raw Data — filenames and modification times.

The legacy `app_shell.py` contains more ambitious analysis—SHAP, model comparisons,
manager/referee/formation summaries, head-to-head views, calibration, and
time-series CV—but it is a 139 KB import-time application with EPL assumptions.
Those ideas should be reimplemented as small core components reading versioned
artifacts; the legacy shell should not be revived as an architecture.

## Root causes and required P0 corrections

### A01 — Team identity is a loose string

**Evidence:** Belgium resolves only 4/18 upcoming clubs exactly. Prediction state
stores are keyed by display strings.

**Consequence:** cold-start priors are silently used for established teams, H2H and
form split across aliases, weather joins miss stadiums, and standings can duplicate a
club.

**Correction:** introduce canonical `team_id`, provider aliases with validity dates,
fuzzy suggestions that never auto-accept in CI, and a hard coverage gate. See the
`EntityResolver` in `03-data-platform-implementation.md`.

### A02 — Team state is split by match role

`HomeTeamPointsLast5` groups only past home matches and `AwayTeamPointsLast5` only
past away matches. `HomeRestDays` and `AwayRestDays` have the same problem. A club's
away match therefore does not update the state used for its next home match.

**Correction:** convert every match into a two-row team-event ledger, calculate
chronological prior state once, then join it back as home and away snapshots.

### A03 — Season identity uses calendar year

`Season = MatchDate.dt.year` splits a normal August–May season in January. Cumulative
points and season goal difference reset at the wrong time.

**Correction:** configure season start month and derive a stable `2026-27` season ID.
Competition editions must be explicit entities, not inferred ad hoc by features.

### A04 — Time handling ignores league configuration

`fetch_upcoming_fixtures.py` converts every kickoff to `US/Eastern`, while
`DataSourceConfig.weather_timezone` is league-specific. The artifact does not preserve
the original UTC timestamp.

**Correction:** store `kickoff_utc`, render in a user-selected timezone, and retain
competition-local time only as a derived display field.

### A05 — Configured competition rules are not used by standings

Belgium's config contains points halving and three split pools; Scotland contains a
top/bottom split. `_standings` simply sums all historical rows and never calls the
phase helpers or applies sanctions.

**Correction:** a rule engine must own table calculation, tie-breakers, phase
transitions, incomplete-season state, and projection labels.

### A06 — H2H and form do not represent full team history

The H2H helper counts wins only when the current home team was also home and the
current away team was also away. Role-reversed wins are omitted. It also treats five
old meetings as equally useful regardless of age or squad turnover.

**Correction:** canonical team IDs, perspective-normalized events, exponential
recency weighting, and a sample-quality label.

### A07 — Cold-start is hidden

Poisson deliberately has empirical priors, but prediction artifacts do not include
how many observations informed each team rate. In Belgium, a generic 1.4118–1.4118
fixture can look like a precise team forecast.

**Correction:** every forecast includes `home_history_n`, `away_history_n`,
`prior_weight`, `entity_resolution_status`, and an uncertainty/cold-start badge.

### A08 — The production gate is too narrow

The audit correctly beats a rolling class-prior baseline on log loss and Brier, but
then selects one candidate globally. It does not test promoted clubs, early-season
fixtures, long-rest fixtures, split phases, favorite/underdog bands, or time decay.

**Correction:** contextual release gates, champion/challenger registry, minimum draw
quality, calibration bounds, and paired bootstrap uncertainty.

### A09 — Model diagnostics are generated but barely exposed

Belgium ships SHAP images, SHAP importance CSV, feature inventory, model ablation,
metadata, and model performance. The Model Lab reads only
`poisson_metrics_history.csv` and renders three metrics plus a table.

**Correction:** generate structured JSON/Parquet diagnostic artifacts and render
reliability, rolling performance, cohort performance, feature drivers, and drift.

### A10 — Artifact versioning is file-level, not domain-level

The manifest validates bytes and a global schema version. It cannot say which
fixture schema, simulation model, entity registry, competition rules, or forecast
contract created a page.

**Correction:** artifact manifest v3 adds typed dataset schemas, dependencies,
row/time coverage, model/rules IDs, and freshness service-level objectives.

### A11 — Package version has two sources of truth

`pitch_oracle_core/_version.py` reports `1.3.27`, while `pyproject.toml` declares
`1.3.26`. Consumer manifests use the former; built wheel metadata uses the latter.

**Correction:** make the build backend read one version source and add a CI assertion
that runtime, wheel metadata, tag, and manifest agree.

### A12 — Data pipeline modules execute work at import time

`prepare_model_data.py` loads files and builds features at module import. This makes
unit testing individual transforms difficult and encourages broad exception handling.

**Correction:** pure transforms, typed inputs, an explicit pipeline command, and
stage-level reports. Compatibility entrypoints can delegate to the new pipeline.

## Keep, refactor, retire

### Keep and extend

- thin consumer repositories and immutable core pins
- strict cache integrity and league binding
- `LeagueConfig` as the entry point for league variation
- chronological splits and point-in-time exclusion policy
- probability-first evaluation (log loss, Brier, calibration)
- optional-provider behavior and cache-first Streamlit runtime
- Poisson goal-market primitives and conservative no-odds fallback

### Refactor behind compatibility shims

- `prepare_model_data.py` into a team-event feature pipeline
- `train_models.py` into a registry-driven training/evaluation command
- `ui_pages.py` into page modules and reusable components
- standings into a rule-aware competition service
- odds into timestamped market snapshots
- model diagnostics into structured artifacts

### Retire after parity

- the monolithic `app_shell.py`
- import-time pipeline execution
- calendar-year seasons
- team display names as join keys
- silent generic priors
- duplicated static SHAP PNGs as the primary explanation format
- hard-coded `US/Eastern` fixture conversion

## Target quality bar

No new analytical page should ship until these gates pass for every consumer:

- 100% current fixture team IDs resolve to a canonical team or are explicitly marked
  as a new club with a documented prior.
- 100% fixtures have a valid UTC kickoff and competition edition.
- standings reproduce an independently checked table for the current phase.
- all historical features prove `feature_timestamp < kickoff_utc`.
- forecast rows expose effective sample size and cold-start status.
- champion beats a season-aware baseline on log loss and Brier, with no material
  calibration regression in defined cohorts.
- every artifact reports schema, producer version, rules version, dependencies,
  row count, coverage interval, generated time, and freshness status.

