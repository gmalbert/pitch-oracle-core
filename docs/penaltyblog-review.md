# penaltyblog Review and Incorporation Recommendation

**Review date:** 2026-09-11  
**Repository reviewed:** `gmalbert/pitch-oracle-core` at `eb7750c`  
**Upstream reviewed:** [`martineastwood/penaltyblog`](https://github.com/martineastwood/penaltyblog), version `1.12.0`

## Executive verdict

Yes, Pitch Oracle should continue incorporating penaltyblog, but as a bounded
statistical and football-data engine rather than as the architecture for the
whole product.

The important fact is that adoption has already happened in this branch. The
repository pins `penaltyblog==1.12.0`, routes ClubElo and football-data.co.uk
ingestion through penaltyblog, uses Dixon-Coles and challenger goal models,
wraps implied odds and metrics, and contains MatchFlow, ratings, xT, Bayesian,
market-grid, betting, and drift modules. The next opportunity is therefore not
another broad migration. It is to make the existing boundary safer, remove
duplicated probability semantics, and promote only the pieces that demonstrate
out-of-time improvement.

The recommended posture is:

1. **Keep penaltyblog at the core boundary** for goal distributions, common
   market calculations, ratings, implied odds, and structured event processing.
2. **Keep Pitch Oracle as the owner** of entity identity, point-in-time
   features, artifact manifests, model promotion, multi-consumer capability
   flags, UI contracts, and responsible staking policy.
3. **Do not treat every upstream feature as production-ready.** Bayesian
   uncertainty, xT, MatchFlow ingestion, scrapers, and alternate models need
   their own data and evaluation gates.

The local test suite passed **292 tests** during this review. The existing
penaltyblog-specific tests are useful smoke and contract checks, but several
are intentionally shallow: they verify imports, output keys, finite values,
or one synthetic fit. They do not yet prove point-in-time safety, posterior
interval correctness, source coverage, artifact provenance, or model promotion
quality. That is why the implementation blueprint adds those test categories.

## What upstream provides today

The upstream package is broader than the original roadmaps assumed:

| Upstream capability | Practical value to Pitch Oracle | Relevant upstream reference |
|---|---|---|
| Poisson, Dixon-Coles, bivariate Poisson, negative-binomial, Weibull-copula, and zero-inflated goal models | Challenger distributions and a fast production baseline | [`models` overview](https://penaltyblog.readthedocs.io/en/latest/models/overview.html) |
| `FootballProbabilityGrid` | One exact-score surface from which 1X2, totals, BTTS, Asian handicap, DNB, expected points, and scoreline probabilities can be derived | [`FootballProbabilityGrid`](https://penaltyblog.readthedocs.io/en/latest/models/football_prob_grid.html) |
| MatchFlow | Lazy processing for nested JSON, StatsBomb, Opta, JSONL, and local event archives | [`MatchFlow`](https://penaltyblog.readthedocs.io/en/latest/matchflow/index.html) |
| ClubElo, football-data.co.uk, Understat, and FBRef scrapers | Standardized source adapters and team-name normalization | [`Scrapers`](https://penaltyblog.readthedocs.io/en/latest/scrapers/index.html) |
| Elo, Massey, Colley, and Pi | Independent rating baselines and form/ranking features | [`Ratings`](https://penaltyblog.readthedocs.io/en/latest/ratings/index.html) |
| Implied odds and betting helpers | Consistent de-vigging, value, Kelly, and arbitrage primitives | [`Implied odds`](https://penaltyblog.readthedocs.io/en/latest/implied/index.html), [`Betting`](https://penaltyblog.readthedocs.io/en/latest/betting/index.html) |
| Proper scores and backtesting | Evaluation and historical strategy experiments | [`Metrics`](https://penaltyblog.readthedocs.io/en/latest/metrics/index.html), [`Backtest`](https://penaltyblog.readthedocs.io/en/latest/backtest/index.html) |
| xT and `viz.Pitch` | Event-derived style and progression features plus optional visualizations | [`Expected Threat`](https://penaltyblog.readthedocs.io/en/latest/xt/index.html), [`Visualizations`](https://penaltyblog.readthedocs.io/en/latest/viz/index.html) |
| Bayesian and hierarchical Bayesian models | Uncertainty experiments, especially for sparse leagues | [`Bayesian models`](https://penaltyblog.readthedocs.io/en/latest/models/bayesian.html) |

The upstream project is MIT-licensed and currently declares Python 3.10–3.13
support. Its project metadata also declares a relatively large dependency
surface, including SciPy, Plotly, Matplotlib, StatsBomb tooling, PuLP, and
network/file-system helpers. That is a reason to keep dependency installation
and import boundaries deliberate, not a reason to remove the package.

## Current incorporation status

| Area | Current state in this repository | Assessment |
|---|---|---|
| Dependency | `penaltyblog==1.12.0` is pinned in `pyproject.toml` and `requirements.txt`; CI covers Windows, macOS, and Linux on Python 3.12/3.13 | Good foundation; lock transitive versions for repeatable builds |
| Historical ingestion | `combine_raw_data.py` uses `FootballData`; `fetch_clubelo.py` uses `ClubElo` | Keep; add snapshot/provenance and point-in-time tests |
| Understat | Adapter exists, but league coverage is capability-gated and the local data-source decision correctly rejects unsupported leagues | Keep as an optional EPL/big-five source, never as a global dependency |
| Goal models | Dixon-Coles with time decay plus several penaltyblog challengers are wrapped in `goal_models.py` and `model_variants.py` | Keep; route production selection through the existing registry and rolling-origin evaluation |
| Probability grids | `markets/grid.py` wraps `FootballProbabilityGrid`, while `domain/probability_grid.py` and `goal_markets.py` implement parallel score/market semantics | Consolidate behind one internal contract before expanding market coverage |
| Implied odds | `pitch_oracle_core/implied.py` uses penaltyblog; `markets/devig.py` retains an internal provider-neutral implementation | Good separation, but document why each method is used and test parity |
| Metrics | `evaluation/baseline.py`, `walk_forward.py`, and the newer evaluation package use penaltyblog and internal score functions | Keep one canonical score implementation; do not let metric names hide different normalization conventions |
| Ratings | Elo, Massey, Colley, and Pi wrappers exist | Useful as lagged features and diagnostics; fix entity coverage and chronological contracts |
| MatchFlow | Local StatsBomb/Wyscout wrappers exist and are exercised by module tests | Not yet a complete production artifact pipeline; add metadata joins, schema validation, and manifest registration |
| xT | Fit/score wrapper exists | Valuable as a feature family, not a standalone match-winner predictor; require competition/season fit IDs |
| Bayesian | Fit wrapper exists | The current uncertainty feature is explicitly heuristic and must not be labeled a credible interval |
| Betting | penaltyblog helpers coexist with Pitch Oracle exposure and staking policy | Keep the upstream math; keep product safety and caps in Pitch Oracle |
| Backtesting | The upstream backtest module is not the system of record | Prefer the existing forecast ledger and rolling-origin framework; use upstream backtest only as an experimental adapter |
| Visualization | A penaltyblog theme wrapper exists, but the core UI still owns the display contract | Optional; do not make the UI depend directly on provider-specific plotting objects |
| FPL | Available upstream but not relevant to most league consumers | EPL-only, capability-gated, and lower priority |

## Findings that should change the existing roadmap

The existing `docs/penaltyblog-roadmaps/` set is useful historical context, but
its “adopt everything” framing is now too broad. In particular:

### 1. The migration is mostly complete, so the next work is integration quality

The old roadmap describes replacing files that have already been replaced or
wrapped. Continuing to follow it literally would duplicate code and obscure the
actual production risks. This review should be the current decision document;
the older roadmaps should be treated as design background.

### 2. `aic_leaderboard.py` is not an AIC leaderboard yet

`pitch_oracle_core.model_variants.aic_leaderboard()` says that penaltyblog 1.12.0
does not expose `.aic`, then emits `n_params` and sorts by parameter count. That
is a fit inventory, not AIC and not evidence that one model forecasts better.
Use the fitted model's actual likelihood/parameter metadata when available, and
label the result `fit_diagnostics` until the calculation is corrected. Model
promotion must use rolling-origin proper scores instead of in-sample AIC alone.

### 3. The walk-forward helper has two correctness hazards

`walk_forward_evaluate()` defaults to `date_col="datetime"`, while
`goals_frame_from_historical()` returns a `date` column. The CLI calls the
helper without overriding the column. The fold range also omits a final fold
when the remaining data is exactly one horizon long. Finally, rows skipped by
`KeyError` or `ValueError` are not reflected in `n_test`, which can make a fold
look complete when it was only partially scored.

### 4. Bayesian uncertainty is currently a placeholder

`uncertainty_features()` sets `bayes_p_home_ci_width` to
`abs(p_home - 0.5) * 0.2` and does not use `n_samples`. This is not derived from
the posterior and can be directionally wrong. Rename the field to a heuristic
if it remains exploratory, or implement a version-tested posterior-predictive
adapter before exposing it in Model Lab or as an ensemble feature.

### 5. Reproducibility metadata is incomplete

The goal-model training hash omits the date column even though dates determine
time-decay weights. It also does not record the penaltyblog version, optimizer
settings, Python/platform information, or the feature policy that produced the
frame. A model artifact can therefore appear reproducible while having been
trained with different recency weights or dependency behavior.

### 6. Ratings need a point-in-time contract

`build_combined_rankings()` derives the team list from home teams only, which
can omit a team that appears only as an away team in a partial frame. More
importantly, full-season Massey/Colley outputs are descriptive rankings and
must not be joined to historical fixtures as if they were pre-match features.
Elo/Pi histories and rolling ratings need explicit “known at” timestamps.

### 7. Two probability-grid implementations are now competing

`FootballProbabilityGrid` is excellent for market extraction, while
`domain.probability_grid.ProbabilityGrid` explicitly models truncation mass and
is the better fit for artifact/audit contracts. The product should choose one
internal representation and put a thin penaltyblog conversion adapter at the
boundary. Otherwise 1X2, tails, expected goals, BTTS, and scoreline semantics
will slowly diverge between pages.

## Recommended adoption order

### P0 — correctness and auditability

1. Add the boundary adapter described in
   [`penaltyblog-implementation-blueprint.md`](penaltyblog-implementation-blueprint.md).
2. Fix walk-forward date fallback, final-fold inclusion, and scored-row counts.
3. Replace the fake Bayesian interval with a real posterior-predictive feature,
   or mark it experimental and keep it out of production artifacts.
4. Correct the AIC report or rename it to fit diagnostics.
5. Include dates, feature-policy version, penaltyblog version, and optimizer
   configuration in training hashes and artifact metadata.
6. Fix ratings entity coverage and require a chronological input contract.

### P1 — highest-value analysis

1. Make the existing rolling-origin registry the promotion authority for every
   penaltyblog model variant.
2. Use `FootballProbabilityGrid` through the internal grid adapter to generate
   one auditable market surface per fixture.
3. Add market-implied expected-goals comparators using penaltyblog's goal-
   expectancy helpers, while retaining bookmaker prices and de-vig method in
   the artifact.
4. Add a closing-line-value and market-consensus report to Model Lab.
5. Add xT and rating features only as lagged, source-scoped feature families.

### P2 — optional intelligence

1. Fit hierarchical Bayesian models only for sparse leagues and compare them
   against the champion with the same folds.
2. Add FBRef/StatsBomb/Opta event features where the consumer has coverage and
   legal access.
3. Add FPL and upstream betting backtests only to an EPL or research consumer.
4. Adopt `viz.Pitch` only if it reduces UI code without coupling the shared
   package to a provider-specific rendering contract.

## What should not be incorporated blindly

- **Understat everywhere:** upstream documents a finite competition list; it is
  not a universal source for the five-league consumer family.
- **Raw live scraping in Streamlit:** pages should read immutable artifacts,
  not trigger provider calls or inherit rate limits.
- **Full-season ratings as historical features:** this is temporal leakage.
- **Bayesian intervals without posterior diagnostics:** uncertainty is a model
  output, not a decorative confidence number.
- **AIC as the promotion gate:** use chronological proper scores, calibration,
  cohort behavior, and operational checks.
- **Kelly as a recommendation by itself:** staking remains disabled unless
  freshness, calibration, uncertainty, executable quote, and exposure gates
  pass.
- **Provider-specific names in the shared domain:** normalize into the existing
  entity registry and preserve source aliases for auditability.

## Bottom line

The strongest use of penaltyblog in Pitch Oracle is as a fast, well-scoped
engine inside a stricter product system. It already supplies enough value to
justify the dependency. The highest-return next step is not adding more modules;
it is making every adopted output traceable, leakage-safe, and comparable to
the existing champion.
