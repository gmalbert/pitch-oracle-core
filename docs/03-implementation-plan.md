# Implementation Plan

> Historical planning document. Phase 0 is complete in core 1.3.0; use
> `consumer-migration-1.3.md` and `templates/consumer` for current onboarding.

## Reality check on the dates first

Today is July 29, 2026. Scotland kicks off in 2 days, Eredivisie/Portugal/Belgium in
~9–12 days, Turkey in ~16–19 days. **Hitting a fully-featured, launch-day app for all
five is not realistic**, especially if Phase 0 (core extraction) is done properly
first. Two honest paths:

- **Fast path:** skip Phase 0, copy/paste `premier-league` for one league to catch
  opening weekend, accept the maintenance debt from `01-architecture-decision.md`,
  and do the proper extraction later as a refactor. Reasonable only if being live
  on day one matters more than long-term maintainability.
- **Right path (recommended):** build `pitch-oracle-core` first, ship the first thin
  league repo 1–2 weeks into its season (results/odds data is retroactively complete
  either way — nothing is lost by not tracking round 1 live), then land the rest on a
  rolling basis through August. This is the plan below.

## Phase 0 — Extract `pitch-oracle-core` (est. 1–1.5 weeks)

1. New repo `pitch-oracle-core`. Move in the modules listed in
   "What moves into `pitch-oracle-core`" (01-architecture-decision.md).
2. Parameterize every hardcoded EPL assumption you find along the way:
   `Div='E0'` → `league_code` arg, ESPN `eng.1` → `espn_slug` arg, the referee/team
   name mapping dicts → loaded from a per-league config file instead of inlined.
   This is mechanical but is the actual point of the phase — do it thoroughly rather
   than fast, since every shortcut here is a shortcut 5 repos inherit.
3. Re-point `premier-league` itself at `pitch-oracle-core` as its first consumer.
   This is your integration test: if EPL predictions still match pre-refactor output
   (same test suite — `test_app_integration.py`, `test_poisson_evaluation.py`,
   `test_precomputed_data.py` — should still pass), the extraction is sound.
4. Tag `pitch-oracle-core v1.0.0`.
5. Spike the two open questions from the data matrix: referee/injury scraper
   coverage, and the Understat-replacement decision. Bake the decision into core
   once, not per league.

## Phase 1 — Pilot league: Eredivisie (est. 3–4 days once core exists)

Recommended as the pilot because it's structurally the simplest of the five
(straight double round-robin, no split/playoff logic — see
`04-per-league-notes.md`), which means Phase 1 tests the *packaging* pattern without
also debugging league-specific structural logic at the same time.

1. New thin repo, e.g. `eredivisie-predictions`.
2. `config.py`: `Div='N1'`, ESPN slug `ned.1`, ClubElo `NED_1`, stadium/team lookup
   table (new build).
3. Pull football-data.co.uk historicals back to 1993/94 via `combine_raw_data.py`
   (now a core function).
4. Wire ClubElo + API-Football. Skip Understat per the core decision. Referee/injury
   scrapers only if the Phase 0 spike confirmed a working source.
5. Train initial ensemble/NN/LSTM models via core's `train_models.py`, run
   `evaluate_poisson.py` to sanity-check goal predictions against a holdout season.
6. Deploy: new Streamlit Community Cloud app, new Cloudflare-routed domain/subdomain
   (your call whether these get their own domains like granitestateappeals.com/
   strictscrutiny.com or live as subdomains of pitch-oracle.com — worth deciding
   once, here, since it sets the pattern for the other four).
7. This is also where you validate the GitHub Actions nightly-update pattern
   transfers cleanly (cron schedule needs adjusting for Eredivisie's Aug–May season
   vs. any different in-season windows).

## Phase 2 — Scotland + Belgium in parallel (est. 3–4 days each)

Both need the split/playoff-phase field designed in `04-per-league-notes.md` before
training starts, so it makes sense to design that schema addition to
`pitch-oracle-core` once and have both repos consume it, rather than doing Scotland,
then rediscovering the same problem for Belgium.

1. Add a `phase` field and phase-aware standings logic to `pitch-oracle-core`
   (used by both leagues, differently configured).
2. Scotland: `Div='SC0'`, ESPN `sco.1`, 12-team/33-game/top-6-bottom-6 split
   config.
3. Belgium: `Div='B1'`, ESPN `bel.1`, Champions/Europe/Relegation play-off config
   (and flag the 2026–27 expansion to 18 teams as a mid-project schema change to
   watch for — see per-league notes).

## Phase 3 — Portugal (est. 2–3 days)

Structurally simple like Eredivisie (no split format), so this should be the fastest
of the four once core + the Phase 2 phase-handling exists (Portugal doesn't need it,
but by Phase 3 the packaging pattern is well-worn). `Div='P1'`, confirm ESPN slug
before starting.

## Phase 4 — Turkey (est. 3–4 days)

Same structural simplicity as Portugal, but budget extra time for:
- Confirming the ESPN fixtures slug (unverified as of this plan)
- Thinner football-data.co.uk odds coverage — decide how the model handles
  leagues with sparser bookmaker columns (fewer books to average/median across)
- A `points_adjustment` field for historical sanction-driven deductions, so
  standings/training data isn't silently wrong for affected seasons

## Phase 5 — Hardening (ongoing, after all 5 are live)

- Backfill referee/injury features for any league where the Phase 0 spike didn't
  clear them for launch, once real sources are confirmed
- Cross-league dashboard/index page linking all 6 (EPL + 5 new) properties if you
  want a portfolio landing page
- Decide on a shared prediction-tracking rollup (`track_predictions.py` already
  exists per-league; a cross-league accuracy leaderboard is a natural v2 feature
  once you have a season of data on all of them)

## Suggested build order summary

| Order | League | Why here |
|---|---|---|
| 0 | Core extraction | Blocks everything else |
| 1 | Eredivisie | Simplest structure, validates the packaging pattern |
| 2 | Scotland + Belgium | Share new split/playoff schema work |
| 3 | Portugal | Simple, fast once pattern is proven |
| 4 | Turkey | Simple structure but needs the most data-quality care |
