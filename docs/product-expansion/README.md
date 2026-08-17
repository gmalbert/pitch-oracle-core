# Pitch Oracle product expansion blueprint

This suite is the implementation plan for turning Pitch Oracle from seven mostly
tabular league pages into a reusable European-soccer intelligence platform. It is
grounded in a code and artifact review of:

- `pitch-oracle-core` at the current `codex/additional-features` working tree
- `belgium-soccer`, the thin Belgian Pro League consumer pinned to core `v1.3.27`
- the Belgium artifact set generated on 2026-08-11

The catalog contains **50 product features** plus **26 evidence-backed research
initiatives**. Every feature is league-neutral:
competition rules, provider availability, naming, local time, season boundaries,
and split/playoff behavior are supplied by configuration or artifacts rather than
hard-coded page branches.

## Read this in order

1. [Repository audit](01-repository-audit.md) — what exists, what is reusable, and
   the concrete failures visible in the Belgium example.
2. [Feature catalog](02-feature-catalog.md) — 50 new capabilities with inputs,
   fallbacks, priority, and acceptance criteria.
3. [Data and platform implementation](03-data-platform-implementation.md) —
   canonical entities, time-aware features, provider capabilities, artifact v3,
   and copy-ready Python contracts.
4. [Modeling implementation](04-modeling-implementation.md) — Elo, Dixon-Coles,
   ensembles, uncertainty, simulation, backtesting, and monitoring code.
5. [UI implementation](05-ui-implementation.md) — the new information
   architecture and reusable Streamlit/Plotly components.
6. [Delivery roadmap](06-delivery-roadmap.md) — ordered work packages, code map,
   migrations, tests, release gates, and rollout across all consumers.
7. [Optional context and markets](07-context-and-market-implementation.md) —
   point-in-time squad/manager/referee context, travel, odds, fair prices,
   responsible staking, and backtesting code.
8. [Open-source landscape](08-open-source-landscape.md) — what mature football
   analytics repositories do well, what to benchmark or borrow, and what not to copy.
9. [Research and betting literature](09-research-and-betting-literature.md) — primary
   papers on score models, dynamic strength, xG/action value, calibration, market
   probabilities, staking, and data snooping, translated into adoption decisions.
10. [Research implementation](10-research-implementation.md) — copy-ready model/grid
    protocols, distribution diagnostics, proper scores, rolling evaluation, calibration,
    de-vigging, exact settlement, provider capabilities, tests, and staged experiments.

## Product thesis

Pitch Oracle should answer five questions for every fixture:

1. **What is likely?** Calibrated 1X2, scoreline, totals, BTTS, and interval forecasts.
2. **Why?** A short evidence chain based only on information available before kickoff.
3. **What could change it?** Lineup, rest, weather, and tactical scenario controls.
4. **What does it mean for the season?** Table, title, qualification, playoff, and
   relegation consequences under the league's actual rules.
5. **Can I trust it?** Freshness, entity coverage, cold-start status, calibration,
   drift, and historical performance in the same context.

## Architectural stance

The thin-consumer architecture is worth keeping. League repositories should remain
small and should not acquire custom analysis code. The change is to make core own a
larger, explicit product contract:

```text
league config + provider adapters
              │
              ▼
canonical teams, competitions, fixtures, snapshots
              │
              ▼
point-in-time feature store + model registry
              │
              ├── match forecasts and explanations
              ├── team and league analytical marts
              ├── season simulation distributions
              └── quality, calibration, and drift reports
              │
              ▼
artifact manifest v3 → shared Streamlit pages
```

The app remains cache-first. Scheduled workflows do expensive ingestion, fitting,
simulation, and explanation generation; Streamlit reads compact Parquet/JSON
artifacts and performs only filtering, light scenario calculations, and rendering.

## Priority summary

| Wave | Outcome | Feature IDs |
|---|---|---|
| P0: trust the inputs | Fix entity, time, season, standings, cold-start, and artifact contracts | F37–F43, F46–F48 |
| P1: make match pages useful | Rich match center, score grid, drivers, trends, uncertainty, and share cards | F01–F12 |
| P2: understand teams and seasons | Team dashboards, strength ratings, fixture difficulty, projections, and rule-aware simulators | F13–F25 |
| P3: improve forecasts | Dynamic goals, ensembles, transfer learning, reconciliation, and model governance | F26–F36 |
| P4: optional intelligence | Odds, squads, managers, referees, weather, alerts, and portfolio tools | F39–F45, F49–F50 |

P0 is not cosmetic cleanup. Without it, Belgium visibly produces generic forecasts
for mismatched team names and a phase-unaware table. Shipping richer visualizations
on top of that would make the product more persuasive without making it more correct.

## Definition of “universal”

A feature is universal when its **contract and fallback behavior** work in every
configured European league. It does not mean every provider must cover every league.
For example, lineup impact is universal because all leagues expose the same
`AvailabilitySnapshot` contract; a consumer without a squad provider renders an
honest “not available” state and the base model continues to work. No page imports a
Belgium-, EPL-, or provider-specific module.
