# Expanding Pitch Oracle to 5 New Leagues

> **New:** the current core-and-Belgium product audit, 50-feature catalog, 26 research
> initiatives, open-source and primary-paper review, copy-ready data/model/UI/research
> implementation, and delivery plan live in
> [`product-expansion/README.md`](product-expansion/README.md). The documents below
> explain the original shared-core extraction and remain useful historical context.

**Scope:** Scottish Premiership, Eredivisie, Primeira Liga, Belgian Pro League, Süper Lig
**Source repo:** `gmalbert/premier-league` ("Pitch Oracle," live at pitch-oracle.com)
**Date:** July 29, 2026

## Why `premier-league` is the base

Of your ~20 repos, `gmalbert/premier-league` is by a wide margin the most mature European
soccer asset:

- 271 commits, MIT-licensed, live in production at pitch-oracle.com
- Ensemble ML (XGBoost + Random Forest + Gradient Boosting + Logistic Regression via soft
  voting), a PyTorch neural net, an LSTM time-series model, and Poisson goal-prediction
  diagnostics — with a model comparison view in the app
- 8 integrated data sources: football-data.co.uk historicals, ESPN fixtures, Understat xG,
  ClubElo ratings, API-Football, weather, scraped referee assignments/stats, scraped
  injury reports
- Supporting infra: `precompute_database.py`, GitHub Actions automation, CI tests
  (`test_app_integration.py`, `test_caching.py`, `test_poisson_evaluation.py`,
  `test_precomputed_data.py`), a PDF report generator, prediction tracking, and its own
  `docs/roadmap-*.md` planning structure
- `mls-predictions` is your other mature soccer repo, but it's built around MLS-specific
  quirks (salary cap, conference structure, turf/travel) that don't transfer — `premier-league`
  is the right architectural donor for four more **European domestic league** builds.

## Bottom line recommendation

**Don't copy/paste the repo 5 times.** Extract the league-agnostic ~70% of
`premier-league` into a shared internal package, and give each new league a **thin repo**
(config + league-specific quirks + Streamlit entrypoint) that depends on that package. Full
reasoning in `01-architecture-decision.md`.

## Documents in this set

1. **01-architecture-decision.md** — copy/paste vs. shared-core vs. monorepo, with a
   recommendation and the tradeoffs of each
2. **02-data-source-matrix.md** — which of `premier-league`'s 8 data sources actually
   cover each of the 5 new leagues, and what breaks
3. **03-implementation-plan.md** — phased build-out, in the order you should actually do it
4. **04-per-league-notes.md** — structural quirks (playoffs, split formats, sanctions) each
   league needs baked into the schema, one section per league
5. **new-consumer-repository.md** — executable bootstrap, GitHub setup, baseline
   source policy, artifact workflow, acceptance gates, and core upgrades for every
   new thin consumer repository
