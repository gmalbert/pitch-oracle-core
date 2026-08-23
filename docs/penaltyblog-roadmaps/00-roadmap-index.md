# penaltyblog Roadmaps for Pitch Oracle

This suite of roadmaps describes every meaningful way the
[`penaltyblog`](https://github.com/martineastwood/penaltyblog) library can be
leveraged inside `pitch-oracle-core`. Every roadmap is self-contained, contains
runnable code, and ends with a "Pitch Oracle integration" section that names
the exact scripts, modules, or scheduled jobs the change touches.

> All code assumes `pip install penaltyblog` (Cython build) and works with
> the public API as of `penaltyblog >= 1.x`. Where private helpers are used,
> they are noted explicitly so they can be replaced if the upstream signature
> drifts.

## Quick orientation

`penaltyblog` is built around **eleven cohesive modules**. Each module solves
a specific part of the football-analytics stack and exposes a small,
well-typed public surface:

| Module              | What it does                                              | Pitch Oracle touchpoints                                     |
|---------------------|-----------------------------------------------------------|--------------------------------------------------------------|
| `penaltyblog.scrapers` | Scrape FBRef, Understat, football-data.co.uk, ClubElo   | `fetch_clubelo.py`, `fetch_understat_xg.py`, `fetch_player_data*` |
| `penaltyblog.matchflow` | Lazy streaming JSON pipelines (StatsBomb, Opta, custom) | `explore_statsbomb.py`, `explore_wyscout.py`                |
| `penaltyblog.models` | Poisson, Dixon-Coles, Bivariate Poisson, etc.             | `train_models.py`, `evaluate_poisson.py`, `optimize_model.py` |
| `penaltyblog.bayes` | MCMC goal models and hierarchical Bayes                   | `model_optimization.py`, `benchmark_hyperparameters.py`      |
| `penaltyblog.ratings` | Elo, Massey, Colley, Pi ratings                          | `fetch_clubelo.py` (replacement), `analyze_team_form.py`     |
| `penaltyblog.implied` | Seven overround-removal algorithms                      | `track_predictions.py`, new `compute_value_bets.py`          |
| `penaltyblog.betting` | Kelly, arbitrage, value-bet detection                    | New `responsible_staking.py`                                |
| `penaltyblog.fpl`   | FPL API scraping + lineup optimizer                       | New `fpl/` workspace for the EPL consumer                    |
| `penaltyblog.xt`    | Expected Threat fit + score                               | New `train_xt_model.py`                                       |
| `penaltyblog.viz`   | Plotly pitch visualizations                              | New `viz/` components inside Streamlit                       |
| `penaltyblog.metrics` | RPS, Brier, log-loss, calibration                        | `evaluate_poisson.py`, new `evaluation/` module, [14-metrics-evaluation.md](14-metrics-evaluation.md) |

A twelfth utility, `FootballProbabilityGrid`, lives inside `models` and turns
any goal model into 30+ derived markets (Asian handicap, totals, BTTS, etc.).

## Read this in order

### Library roadmaps

1. [01-scrapers-data-collection.md](01-scrapers-data-collection.md) — drop-in
   replacements for the hand-rolled scrapers in `pitch-oracle-core` with
   uniform column names and a single team-name normalizer.
2. [02-matchflow-event-pipelines.md](02-matchflow-event-pipelines.md) — replace
   the eager `pd.read_json` calls in `explore_statsbomb.py` and
   `explore_wyscout.py` with a lazy streaming pipeline that scales to
   multi-season event archives.
3. [03-goal-models.md](03-goal-models.md) — replace `evaluate_poisson.py` and
   `train_models.py` with the Cython-accelerated Poisson / Dixon-Coles /
   Bivariate Poisson / Negative Binomial / Weibull Copula / Zero-Inflated
   Poisson family. Same inputs, much faster scoring, plus
   `FootballProbabilityGrid` for derived markets.
4. [04-bayesian-models.md](04-bayesian-models.md) — add the
   `BayesianGoalModel` and `HierarchicalBayesianGoalModel` so that the
   `model_optimization.py` script can also output **full posterior
   distributions** instead of point estimates.
5. [05-ratings-systems.md](05-ratings-systems.md) — use Elo, Massey, Colley,
   and Pi ratings to enrich `analyze_team_form.py` and to compute new
   features (form deltas, league priors, matchup quality).
6. [06-implied-odds.md](06-implied-odds.md) — apply seven overround-removal
   algorithms to bookmaker prices so the "value bet" detector stops being
   fooled by margin.
7. [07-betting-utilities.md](07-betting-utilities.md) — produce Kelly-sized
   stakes, detect multi-bookmaker arbitrage, and surface value bets.
8. [08-fantasy-premier-league.md](08-fantasy-premier-league.md) — stand up
   an FPL workspace inside the EPL consumer (player scraper, optimal
   lineup, captain picks).
9. [09-expected-threat.md](09-expected-threat.md) — fit an xT model from
   StatsBomb open data and use it to score every pass/carry in the pipeline.
10. [10-visualizations.md](10-visualizations.md) — replace Plotly pitch
    snippets inside Streamlit with `penaltyblog.viz.Pitch` and ship a
    consistent visual style across all consumers.
11. [11-probability-grid.md](11-probability-grid.md) — extend any goal model
    with `FootballProbabilityGrid` to produce a unified market surface.
12. [12-statsbomb-opta-connectors.md](12-statsbomb-opta-connectors.md) —
    use the MatchFlow StatsBomb / Opta adapters to stream events straight
    into the model pipeline.

### Cross-cutting roadmaps (Pitch Oracle-specific)

13. [13-integration-with-pitch-oracle.md](13-integration-with-pitch-oracle.md)
    — the consolidation: which scripts get deleted, which jobs get
    re-pointed, what the rollout looks like, and what tests pin the
    migration.
14. [14-metrics-evaluation.md](14-metrics-evaluation.md) — the measurement
    instrument: proper scores, reliability, calibration, decomposition,
    cohorts, the forecast ledger, and the model registry. Closes the
    loop between "we made a forecast" and "we know whether the forecast
    was any good."
15. [15-production-observability.md](15-production-observability.md) —
    the closed loop: input drift (PSI / KS), output drift, calibration
    drift, coverage drift, operational drift, severity routing, and
    regression suites. The thing that catches a model going bad before
    users do.
16. [16-ui-streamlit-pages.md](16-ui-streamlit-pages.md) — the new
    information architecture, the reusable component layer, page
    shells, capability badges, and the Model Lab / Value Bets / Today
    Bets pages. Consolidates the UI changes scattered across roadmaps
    03–11.
17. [17-multi-consumer-patterns.md](17-multi-consumer-patterns.md) —
    league-neutral patterns, capability flags, the thin-consumer
    contract, provider adapters, and the cross-league index.
    Stops the integration from collapsing into an EPL-only design.
18. [18-test-data-reproducibility.md](18-test-data-reproducibility.md) —
    golden-master fixtures, property-based tests, regression suites,
    reproducibility rules, and the "fresh-fixture" workflow. The
    contract between this migration and the next one.

### Synthesis

- [00-priority-list.md](00-priority-list.md) — the prioritised
  execution list, derived from all 18 roadmaps above. P0 / P1 / P2 /
  P3 with effort estimates, dependencies, exit gates, and the
  definition of done. Read this first when you want to know
  *what to do next*.

## Why these roadmaps matter

Pitch Oracle already has a working data pipeline, an ensemble ML stack,
and a polished Streamlit UI. `penaltyblog` does not replace any of that —
it **compresses** three categories of work:

1. **Scraping boilerplate** — hand-rolled parsers for Understat, ClubElo,
   FBRef, and football-data.co.uk become one-line calls with stable column
   contracts.
2. **Statistical modelling** — the in-house Poisson evaluator and Dixon-Coles
   notebook are replaced by battle-tested Cython implementations that handle
   edge cases (neutral venues, weights, convergence).
3. **Market math** — every derived probability (1X2, totals, BTTS, Asian
   handicap, fair odds) is computed consistently from one probability grid,
   so two screens in the UI can no longer disagree about the same fixture.

The cost is a new dependency and an investment in the migration. The
benefit is that every future league consumer inherits the same statistical
engine and the same UI components for free.

## What the cross-cutting roadmaps add

The 12 library roadmaps above describe *what* to do. The five
cross-cutting roadmaps (14–18) describe *how* to do it in Pitch
Oracle's specific context:

- **[14-metrics-evaluation.md](14-metrics-evaluation.md)** is the
  measurement instrument. Without it, every other roadmap is
  guesswork.
- **[15-production-observability.md](15-production-observability.md)**
  is the closed loop. It catches a model going bad before users do.
- **[16-ui-streamlit-pages.md](16-ui-streamlit-pages.md)** prevents
  the new pages from ending up with five different colour schemes
  and three different reliability diagrams.
- **[17-multi-consumer-patterns.md](17-multi-consumer-patterns.md)**
  keeps the integration from collapsing into an EPL-only design and
  enforces the thin-consumer contract mechanically.
- **[18-test-data-reproducibility.md](18-test-data-reproducibility.md)**
  is the contract between this migration and the next one.

## Conventions used in every roadmap

- All code blocks are runnable as long as `penaltyblog` is installed.
- Where a script name is referenced (e.g. `train_models.py`), it points at a
  file that already exists in `pitch-oracle-core/` root.
- "Migration cost" estimates assume one developer familiar with the repo.
- All paths are relative to the `pitch-oracle-core/` working tree.
