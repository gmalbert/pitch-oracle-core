# 13 — Pitch Oracle Integration Plan

This is the consolidation roadmap. It walks through every existing
script in `pitch-oracle-core/`, says what changes, and orders the work
into migration waves that ship independently.

## 13.1 Inventory of existing scripts

| Script                                 | Action                                                  |
|----------------------------------------|---------------------------------------------------------|
| `fetch_clubelo.py`                     | Replace body with `ClubElo` (see [01](01-scrapers-data-collection.md)) |
| `fetch_understat_xg.py`                | Replace body with `Understat`                           |
| `fetch_api_football.py`                | Keep — no penaltyblog equivalent                        |
| `fetch_player_data.py`                 | Partial: keep FPL-specific bits, add `pb.fpl.*` calls   |
| `fetch_player_data_fd.py`              | Partial: keep betting-specific bits                     |
| `fetch_upcoming_fixtures.py`           | Keep, but route through `FootballData`                  |
| `fetch_upcoming_features_espn.py`      | Keep — ESPN is not in `penaltyblog`                     |
| `fetch_weather_data.py`                | Keep — weather is not in `penaltyblog`                  |
| `scrape_injuries.py`                   | Keep — injuries are not in `penaltyblog`                |
| `scrape_injuries_web.py`               | Keep                                                   |
| `scrape_referees.py`                   | Keep                                                   |
| `combine_raw_data.py`                  | Replace with thin orchestrator over the four scrapers   |
| `team_name_mapping.py`                 | Delete (replaced by `pitch_oracle_core/team_mappings.py`) |
| `explore_statsbomb.py`                 | Rewrite as MatchFlow pipeline (see [02](02-matchflow-event-pipelines.md), [12](12-statsbomb-opta-connectors.md)) |
| `explore_wyscout.py`                   | Rewrite as MatchFlow pipeline                           |
| `train_models.py`                      | Replace with `DixonColesGoalModel` (see [03](03-goal-models.md)) |
| `evaluate_poisson.py`                  | Keep diagnostics; swap model class                      |
| `model_optimization.py`                | Add Bayesian tab (see [04](04-bayesian-models.md))     |
| `benchmark_hyperparameters.py`         | Replace optimiser with `minimizer_options`              |
| `optimize_model.py`                    | Replace with `DixonColesGoalModel.fit`                  |
| `compare_model_features.py`            | Add the new models                                     |
| `analyze_team_form.py`                 | Add Elo / Massey / Colley / Pi deltas (see [05](05-ratings-systems.md)) |
| `track_predictions.py`                 | Replace implied-probability code with `pb.implied` (see [06](06-implied-odds.md)) |
| `validate_models.py`                   | Walk-forward + Bayesian calibration (see [04](04-bayesian-models.md)) |
| `precompute_database.py`               | Add new scraper + model outputs                         |
| `precompute_model_diagnostics.py`      | Add walk-forward, calibration, agreement metrics        |
| `generate_pdf_report.py`               | Wire new value-bet section (see [07](07-betting-utilities.md)) |
| `app_factory.py`                       | Add pages for Value Bets, Today's Bets, Expected Threat, FPL |

## 13.2 Migration waves

The work is split into seven waves, each independently shippable.

### Wave 0 — dependencies (½ day)

- `pip install penaltyblog` (Cython build).
- Add `penaltyblog>=1.x` to `pyproject.toml` / `requirements.txt`.
- Verify `python -c "import penaltyblog as pb; print(pb.__version__)"`.

### Wave 1 — scrapers (3 days)

Per [01](01-scrapers-data-collection.md):

- Add `pitch_oracle_core/team_mappings.py`.
- Rewrite `fetch_clubelo.py`, `fetch_understat_xg.py`.
- Add `scripts/ingest/football_data_history.py`.
- Add `scripts/ingest/fbref_team_stats.py`.
- Delete `team_name_mapping.py` once nothing imports it.

Tests:

- `tests/test_scrapers.py` — assert column contracts.
- `tests/test_team_mappings.py` — assert all scrapers normalise a known
  set of team names.

### Wave 2 — event pipelines (4 days)

Per [02](02-matchflow-event-pipelines.md) and [12](12-statsbomb-opta-connectors.md):

- Rewrite `explore_statsbomb.py`, `explore_wyscout.py` as MatchFlow.
- Add `scripts/events/matchflow_shots.py`,
  `scripts/events/matchflow_passes.py`,
  `scripts/events/matchflow_pressures.py`.
- Wire the new pipelines into `precompute_database.py`.

Tests:

- `tests/test_matchflow_pipelines.py` — golden-master the pipeline
  output on a fixture set.

### Wave 3 — goal models (5 days)

Per [03](03-goal-models.md):

- Rewrite `train_models.py`, `evaluate_poisson.py`,
  `optimize_model.py`, `benchmark_hyperparameters.py`.
- Add `scripts/eval/walk_forward.py`, `scripts/eval/aic_leaderboard.py`.
- Add `models/` directory for persisted model artifacts.

Tests:

- `tests/test_dixon_coles.py` — verify fit + predict + grid.
- `tests/test_negative_binomial.py`, `tests/test_bivariate_poisson.py`,
  `tests/test_weibull_copula.py`, `tests/test_zero_inflated_poisson.py`.
- `tests/test_propagation.py` — predict same match with three different
  models and assert monotonicity where expected.

### Wave 4 — Bayesian (4 days)

Per [04](04-bayesian-models.md):

- Add `scripts/eval/bayesian_calibration.py`.
- Add `scripts/ensemble/build_features.py` (uncertainty features).
- Update `model_optimization.py` to add a Bayesian tab.

Tests:

- `tests/test_bayesian_model.py` — verify convergence on a small
  fixture set.
- `tests/test_hierarchical_bayesian.py`.

### Wave 5 — ratings + betting (4 days)

Per [05](05-ratings-systems.md), [06](06-implied-odds.md),
[07](07-betting-utilities.md):

- Add `scripts/ratings/build_rankings.py`.
- Add `scripts/value/daily_value_scan.py`.
- Add `scripts/odds/margin_audit.py`.
- Add `streamlit_pages/06_today_bets.py`.

Tests:

- `tests/test_ratings.py` — Elo update rules, Massey linear system,
  Colley with draws, Pi ratings symmetry.
- `tests/test_implied.py` — seven methods on a known fixture.
- `tests/test_betting.py` — Kelly single + multi, value, arbitrage,
  hedge.

### Wave 6 — xT + FPL + viz (6 days)

Per [08](08-fantasy-premier-league.md), [09](09-expected-threat.md),
[10](10-visualizations.md), [11](11-probability-grid.md):

- Add `scripts/xt/fit_xt.py`, `scripts/xt/score_events.py`.
- Add `fpl/` workspace in the EPL consumer.
- Add `pitch_oracle_core/markets.py`.
- Rewrite Streamlit match-centre and team-profile pages to use
  `Pitch`.

Tests:

- `tests/test_xt_model.py` — fit on synthetic events, verify xT
  surface shape.
- `tests/test_markets.py` — every market helper against golden values.
- `tests/test_viz_pitch.py` — smoke-test Pitch with one provider.

### Wave 7 — productionisation (3 days)

- Wire all artifacts into the Streamlit cache manifest.
- Wire all artifacts into the GitHub Actions schedule.
- Update `pitch-oracle-core/README.md` to reflect new dependencies.
- Update CI to install penaltyblog.

Tests:

- `tests/test_artifacts.py` — every artifact file referenced in the
  cache manifest exists and has the expected schema.

## 13.3 Backwards-compatibility rules

During the migration, the existing Streamlit pages must continue to
work. Two safety rules:

1. **Old artifacts are still valid.** Keep `data_files/raw/*.parquet`
   in its current shape; new artifacts go alongside.
2. **Old scripts keep their CLI.** `python fetch_clubelo.py` should
   still write `data_files/clubelo_today.csv` with the same columns.
   Internally it now uses `pb.ClubElo`, but the on-disk format is
   unchanged.

## 13.4 Test surface

The existing test suite is small (see `tests/`):

```
test_app_integration.py
test_caching.py
test_poisson_evaluation.py
test_precomputed_data.py
```

The migration adds:

```
test_scrapers.py                  # contracts
test_team_mappings.py             # normalisation
test_matchflow_pipelines.py       # golden output
test_dixon_coles.py               # fit + predict + grid
test_negative_binomial.py         # …
test_bivariate_poisson.py         # …
test_weibull_copula.py            # …
test_zero_inflated_poisson.py     # …
test_propagation.py               # model-monotonicity
test_bayesian_model.py            # convergence
test_hierarchical_bayesian.py     # …
test_ratings.py                   # Elo / Massey / Colley / Pi
test_implied.py                   # seven methods
test_betting.py                   # Kelly / value / arb / hedge
test_xt_model.py                  # fit + score + plot
test_markets.py                   # grid market helpers
test_viz_pitch.py                 # Pitch smoke tests
test_artifacts.py                 # cache-manifest sanity
```

That's 21 test files. Most are < 100 lines; the longest is probably
`test_matchflow_pipelines.py` (~250 lines).

## 13.5 Risk register

| Risk                                                          | Likelihood | Impact | Mitigation                                                |
|---------------------------------------------------------------|------------|--------|-----------------------------------------------------------|
| `penaltyblog` Cython build fails on Windows / Apple Silicon  | Medium     | High   | Pin `penaltyblog` to a wheel-published version; CI matrix |
| Scraper rate limiting triggers IP ban (FBRef)                | High       | Medium | Honour `min_request_interval`, cache aggressively         |
| Bayesian MCMC is slow in CI                                  | Medium     | Low    | Run a tiny version (n_samples=200) in CI                  |
| `FootballProbabilityGrid` semantics drift across versions     | Low        | High   | Pin `penaltyblog==1.x.y`; add a smoke test                |
| StatsBomb archive URL changes                                 | Medium     | Low    | Pin a SHA in `data_files/statsbomb/README.md`             |
| MatchFlow API changes                                         | Low        | Medium | Same — pin + smoke test                                   |
| xT model overfits to a single league                          | Medium     | Medium | Always validate on a held-out competition                 |

## 13.6 Estimated total migration cost

| Wave    | Days    | Owner       | Output                                           |
|---------|---------|-------------|--------------------------------------------------|
| 0       | 0.5     | Eng         | Working `import penaltyblog`                     |
| 1       | 3       | Eng         | Scraper parity with old behaviour + tests        |
| 2       | 4       | Eng         | MatchFlow pipelines + tests                      |
| 3       | 5       | Eng + Stats | Goal models + walk-forward eval                  |
| 4       | 4       | Stats       | Bayesian models + calibration dashboard          |
| 5       | 4       | Eng         | Ratings + implied + value-bets                   |
| 6       | 6       | Eng + UI    | xT, FPL, viz, markets                            |
| 7       | 3       | Eng         | CI + actions + docs                              |
| **Total** | **29.5** | | **A polished penaltyblog-powered pitch-oracle-core** |

Roughly six engineer-weeks. After that, `pitch-oracle-core/` becomes a
**thin consumer of penaltyblog**, and every new league repo inherits the
same statistical engine, the same UI components, and the same data
contracts.

## 13.7 What changes for users

The pitch-oracle.com UI gains:

- A "Value Bets" tab with daily opportunities.
- A "Today's Bets" tab summarising the day's recommendations.
- A "Model Comparison" tab with the AIC leaderboard.
- A "Bayesian" tab with credible intervals on every probability.
- An "Expected Threat" tab with the xT heatmap and per-player xT
  rankings.
- An "FPL" tab inside the EPL consumer with optimal lineup + captain
  advisor.

What stays the same:

- The existing fixture detail, team profile, table, and form pages.
- The cached artifact workflow (Streamlit reads parquet, no live
  scraping in the page).
- The Streamlit visual style (until the viz migration in wave 6).

## 13.8 Definition of done

The migration is complete when:

- [ ] Every script in the inventory table has been replaced, deleted,
      or annotated as "kept".
- [ ] Every test in the test surface above passes.
- [ ] CI runs `pip install penaltyblog` and the full test suite on
      Linux + Windows + macOS.
- [ ] `pitch-oracle.com` renders the new "Value Bets", "Today's Bets",
      "Bayesian", "Expected Threat", and (for the EPL consumer) "FPL"
      tabs.
- [ ] The README is updated to mention `penaltyblog` as a dependency.
- [ ] The migration waves 0–7 each have a corresponding PR with tests.

## 13.9 What's next after this migration

Once `penaltyblog` is the default engine, three new opportunities open
up:

1. **Hierarchical cross-league priors.** Train one Bayesian model that
   shares information across leagues — relegation candidates in the
   Championship get a Premier League prior. This is what
   `HierarchicalBayesianGoalModel` is designed for.
2. **In-play markets.** Combine live `Understat` xG with the
   `FootballProbabilityGrid` to compute in-play Asian handicap fair
   prices.
3. **xT-based feature engineering.** The ML ensemble gets a long tail
   of xT-derived features (per-action xT, xT risk, xT by zone). Each
   one is a small win; together they are a step-change in model
   quality.

Those are all future roadmaps; the current scope is the 13 roadmaps
above.
