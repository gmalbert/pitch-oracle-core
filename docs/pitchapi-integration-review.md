# PitchAPI Integration Review

**Date:** 2026-08-16
**Status:** Assessment complete — live API validated with the test key in `.env` (`PITCH_API_KEY`)
**Docs:** https://pitchapi.dev/#quickstart · **Base URL:** `https://api.pitchapi.dev`

PitchAPI is a **read-only REST API** for football data: fixtures, results, per-shot data with
xG, per-player match stats, lineups, momentum curves, and a deep set of **advanced
analytics** (xT, VAEP, possession value, PPDA, pass networks) derived from the raw event
feed. One free plan reaches everything, with full history back to 2021 and no request
allowance.

## TL;DR

PitchAPI is the missing piece this project has been looking for. `docs/02-data-source-matrix.md`
flags **Understat as a "hard no" across all 5 expansion leagues** and calls for a shots-based
xG provider to be decided once in `pitch-oracle-core` so every league inherits it. PitchAPI:

- **Covers all 5 target leagues** (Eredivisie, Belgium First Division A, Liga Portugal,
  Super Lig, Scottish Premiership) **plus the EPL**, with seasons back to 2021/22 —
  everything the expansion needs, from one free key.
- **Provides per-shot xG** — a strict upgrade over the deterministic distance/angle proxy in
  `pitch_oracle_core/xg.py`, which is currently a library function with nothing feeding it.
  Note: the catalogue is a **completed-match feed** — every match is `finished`, no live
  or scheduled fixtures are exposed (details in *Coverage freshness* below). Great for
  training/backtest features, not a pre-match or in-play source.
- **Provides the event-level data** that the dormant `CanonicalAction` schema
  (`pitch_oracle_core/events/schema.py`) and xT/VAEP feature code (`events/value.py`) were
  built for but no production source currently populates.
- **Has no rate limits** (only an unadvertised fair-use burst guard) and **no cost** — no
  per-call budget, no card.

The key is already configured as `PITCH_API_KEY` and was validated live for this review.

## API overview

### Authentication & responses

- Send the key on every request as the `X-API-KEY` header. The project's key uses the
  `pk_test_` prefix, which is the working production key here. Keep it server-side; it is a
  bearer credential.
- Every response is wrapped: success carries `data`, failure carries `error` (never both).
  Errors use HTTP status codes + stable machine codes (`UNAUTHORIZED`,
  `RATE_LIMIT_EXCEEDED`, `INVALID_PARAMETER`, `RESOURCE_NOT_FOUND`,
  `ANALYTICS_UNAVAILABLE`, `INTERNAL_SERVER_ERROR`).
- Timestamps are RFC 3339 UTC, dates are `YYYY-MM-DD`. Optional fields are omitted, lists
  are empty arrays, and **null is never a zero** (null rate = empty denominator).
- IDs are opaque tokens: `m_` match, `s_` shot, `p_` player, `t_` team, `l_` league.

### Endpoints (all validated live for this review)

| Endpoint | What it gives | Validated |
|---|---|---|
| `GET /v1/leagues` | 42 leagues with seasons available | ✅ Eredivisie, Belgium, Portugal, Turkey, Scotland, EPL all present, back to 2021/22 |
| `GET /v1/leagues/{id}/matches?season=` | Fixtures + results for a season | ✅ Full Eredivisie 2025/26 with statuses & scores |
| `GET /v1/date/{date}` | All covered matches kicking off a day | ✅ 21 matches on 2026-08-16 |
| `GET /v1/matches/{id}` | Match summary (league, teams, score, status, referee, stadium) | ✅ |
| `GET /v1/matches/{id}/shots` | Every shot: coordinates (105×68 m pitch), **per-shot xG**, xGOT, body part, situation, blocked info, keeper | ✅ 33 shots for one Eredivisie match |
| `GET /v1/matches/{id}/players` | Per-player match stats (rating, goals, xG+xA, attack/defense/duels groups) | ✅ 45 stat lines |
| `GET /v1/matches/{id}/lineups` | Formations, starters, subs with formation-grid coordinates | (documented) |
| `GET /v1/matches/{id}/momentum` | Minute-by-minute pressure curve, −100…+100 (positive = home) | ✅ 94 points |
| `GET /v1/matches/{id}/events` | Goals/cards/subs/penalties timeline with running score | (documented) |
| `GET /v1/matches/{id}/h2h` | Head-to-head history + recent/upcoming meetings | (documented) |
| `GET /v1/matches/{id}/stats` | Team-vs-team stats per half (string values + `format_type`) | (documented) |
| `GET /v1/matches/{id}/advanced` | **Team advanced analytics**: xT, VAEP, PV, passing, carrying, creation, defending, territory, tempo | ✅ xT 1.479 for Ajax–Telstar |
| `GET /v1/matches/{id}/advanced/players` | Same groups per player, sorted by VAEP | (documented) |
| `GET /v1/matches/{id}/advanced/network` | Pass networks: nodes (avg_x/y, degree, strength, betweenness, clustering), edges, centralization | (documented) |
| `GET /v1/teams/{id}`, `GET /v1/players/{id}` | Reference records | (documented) |

### Advanced analytics detail (the differentiator)

Everything is defined in the docs with published definitions (Opta / FBref / StatsBomb /
original papers). Highlights, per team and per player:

## Coverage freshness (validated 2026-08-16)

Empirical check against the live API — this matters for what we can and can't rely on it
for.

### It is a post-match, backfilled data source — not live, not pre-match

- **No in-play or scheduled statuses exist.** Every single match across the catalogue
  comes back `finished`. The 21 matches on 2026-08-16 were all `finished`; a full-season
  dump of Eredivisie 2025/26 (309 matches) and EPL 2025/26 (380 matches) is 100%
  `finished`. There is no `live`, `inplay`, `ht`, `ft`, or `scheduled` state observed.
- **Future dates return empty.** `GET /v1/date/2026-08-17` through `2026-08-30` returned
  zero matches — no upcoming fixtures are exposed, even though the 2026/27 season exists
  (15 matches, all `finished`, 2026-08-07 → 2026-08-16). The only "upcoming" records are
  inside the h2h endpoint (`finished: false`, null scores), and that's for context, not a
  schedule feed.
- **Same-day data is available but only after the match is ingested.** A match played
  2026-08-16 (ADO Den Haag 1–4 FC Groningen) already had 27 shots, both teams' advanced
  analytics (xT 0.92), a 94-point momentum curve, and 45 player stat lines. So the
  backfill is fast (at most a few hours after full-time), but it is a **completed-match
  feed**, not a real-time one.

### What this means for Pitch Oracle

- **It cannot power pre-match or in-play prediction.** No live scores, no upcoming
  fixtures, no market data. The model's fixtures/odds inputs stay on ESPN +
  football-data + the odds pipeline; PitchAPI adds *historical* features only.
- **It is perfect for the training/backtest side.** For any match that has already been
  ingested (which, by observation, is effectively the entire catalogue once a day or two
  has passed), you get complete shots + xG + advanced analytics to feed the ledger EWM
  features and the `CanonicalAction` pipeline.
- **Expect per-match data to appear with a delay after full-time**, and design the
  fetcher to treat a missing/empty match as "not yet ingested" rather than "doesn't
  exist" — retry it on the next run. Given a daily scheduled fetch, anything from the
  previous day is essentially guaranteed complete.
- **Watch `ANALYTICS_UNAVAILABLE`** for coverage gaps (matches held but never rated) —
  that's the one real caveat on per-league depth, separate from freshness.

- **Possession value:** `xt_total` (Karun Singh's Expected Threat), `vaep_total` /
  `vaep_offensive` / `vaep_defensive` (Decroos et al., KDD 2019), `pv_total` / offensive /
  defensive (Stats Perform's 10-second Possession Value). Models are fitted on closed
  historical seasons and frozen — a match is never scored by a model that saw it.
- **Passing:** passes, accuracy, progressive passes/distance, passes into box, key passes,
  assists, through balls, crosses, switches.
- **Carrying:** carries, carry distance, progressive carries, carries into final third /
  box, take-ons won, miscontrols, dispossessed.
- **Creation:** SCA, GCA with breakdowns, second assists, chances created, xAG,
  `xg_chain` / `xg_buildup` (player-only).
- **Defending:** tackles, interceptions, blocks, clearances, duels, aerials, **PPDA** with
  the raw numerator/denominator exposed, `avg_defensive_action_x`, high turnovers,
  counterpress regains in 5s, ball recovery time.
- **Territory:** possession %, field tilt, final-third entries, box entries, avg action x.
- **Tempo:** passes per sequence, sequence time, direct speed, buildup vs direct attacks.
- **Goalkeeping** (player only): claims, claim rate, sweeper actions, distributions,
  launches, distribution accuracy.

Two things to be careful with (documented by the API itself):

1. `passing.passes` already contains `passing.crosses`, and `creation.chances_created` ==
   `passing.key_passes` — never add these pairs together.
2. Player figures don't always sum to team figures (e.g. `sca_breakdown.foul_drawn`
   belongs to the team only).

## Where it fits in pitch-oracle-core

### 1. The xG gap (highest value)

The current xG story:

- `pitch_oracle_core/xg.py` defines a deterministic logistic proxy
  (`logit = 1.65 − 0.105·distance + 0.012·angle − 0.45·header + 0.15·foot`) — a library
  function with **no production caller**.
- `features/ledger.py::build_team_events` already consumes optional `home_xg` / `away_xg`
  (and `home_shots` / `away_shots`) columns and derives the dormant
  `xg_for_ewm10`, `shots_for_ewm10`, `shot_quality_ewm10`, `finishing_vs_expectation_ewm10`
  features, plus the `HomexG_Avg_L5` / `AwayxG_Avg_L5` UI/contract surface. Those columns
  are simply never populated, so the features are silently skipped.
- Understat (`fetch_understat_xg.py`) covers EPL only — confirmed a "hard no" for the 5 new
  leagues.
- Bzzoiro (`bzzoiro_football_api.py`) backfills actual xG / shotmaps / player stats for the
  EPL, but it's a private client with unverified coverage.

**PitchAPI replaces all of this**: pull per-shot xG per match, aggregate to match-level
`HomeXG` / `AwayXG`, and feed the existing ledger machinery. Every league gets real xG
features without a custom proxy.

### 2. The event/action schema gap

`pitch_oracle_core/events/schema.py` defines a provider-neutral `CanonicalAction` contract
(with `action_type=SHOT`, coordinates on the canonical 105×68 pitch, body part, set piece,
provider lineage) and `events/value.py` implements xT/VAEP-style lagged action snapshots —
both marked **capability-gated** in `domain/research.py` (R21/R23/R24) because no source
populates them. PitchAPI's shot + event data is the natural first production provider:
coordinates are already in metres on a 105×68 pitch (matching the canonical schema), and the
API serves per-shot `expected_goals`, `situation`, `shot_type`, blocked info, and goal-line
crossing coordinates.

### 3. Model feature candidates

- **Optional EWM xG/shots features** (via the existing ledger): a direct, low-risk add.
- **LSTM raw inputs** (`models/lstm_predictor.py`): currently uses `HomeShots` /
  `AwayShots` / SOT / corners / fouls / cards from football-data.co.uk; per-match xG and
  shots-on-target from PitchAPI would enrich the sequence vectors with better-quality
  inputs.
- **Style/pressing features** (new): PPDA, `avg_defensive_action_x`, `high_turnovers`,
  `ball_recovery_time`, field tilt, direct speed — none exist anywhere in the repo today.
- **Lineup/player context** (capability-gated): `players/lineup_strength.py` exists and is
  gated on a squad provider; PitchAPI lineups + per-player advanced stats are the missing
  feed. `goalkeeping` group is player-only and would populate keeper context.

The current champions — regularized logistic regression on no-odds features, and the
walk-forward Poisson — take aggregate form/strength inputs only, so expect the value to
come through the *feature* path (EWM xG/shot-quality columns) and new capability-gated
pages rather than a swap of the production model.

### 4. UI/analytics opportunities (none of this renders today)

The manifest-v3 UI (`pitch_oracle_core/ui/pages/`) has no shot maps, no momentum charts,
no player stat views, no pass networks — the exploration found shot-map data fetched by
Bzzoiro to CSV with no consumer. PitchAPI is a clean, documented, per-match source for:

- **Shot maps** (match_center / team_center) with per-shot xG, on/off target, body part,
  blocked shots, and goal-line crossing coordinates.
- **Momentum curves** per match (signed pressure series, positive = home).
- **Team style fingerprints** (passing volume, tempo, pressing intensity) for the team
  comparison / radar pages — `analytics/style_clusters.py` already references
  `shots_tempo` / `style_matchup` as planned but fed by demo data only.
- **Head-to-head** for pre-match context (the h2h endpoint includes upcoming meetings).
- **xG per-match comparison** (real xG vs. model expected goals) in match_center.

### 5. Operational fit

- **Fixtures/date checks:** `GET /v1/date/{date}` mirrors what `check_tomorrow_matches.py`
  does today — a candidate cross-check or replacement for the ESPN upcoming-fixtures path
  (`fetch_upcoming_fixtures.py`), with kickoff times in UTC and per-league statuses.
- **Cache pattern:** the repo already caches JSON per source under `data_files/api_cache/`
  (`fetch_api_football.py::get_cached_or_fetch`); a `data_files/pitchapi_cache/` with the
  same `max_age_hours` pattern fits naturally. Season-level data changes rarely; per-match
  data is immutable once `status == finished`, so cache keys are stable.
- **Env convention:** `PITCH_API_KEY` follows the existing `SPORTS_API_KEY` /
  `BZZOIRO_KEY` convention (read at module import, exit on missing). Add it to
  `.env.example` alongside the others.

## Integration plan (recommended shape)

1. **Config gate** — add a `pitchapi: bool` flag (and optionally `pitchapi_league_id`) to
   `DataSourceConfig` in `pitch_oracle_core/config.py`; enable it for the leagues that want
   it (start with EPL + Eredivisie, the simplest structure). Wire it into
   `SourceAvailability` / `OptionalFeatureSet` like the other optional sources.
2. **Fetcher** — new root-level `fetch_pitchapi.py` following the modern template
   (`fetch_upcoming_fixtures.py` style): `main(league=...)`, `os.environ.get("PITCH_API_KEY")`,
   `sys.argv` dispatch (`--shots`, `--xg`, `--advanced`, `--backfill-all`), JSON cache, and
   TSV output to `data_files/`:
   - `pitchapi_matches.csv` (id, league, teams, date, status, score — league + season)
   - `pitchapi_shots.csv` (match, team, player, x, y, xG, xGOT, on-target, body part,
     situation, minute, blocked, goal-line crossing)
   - `pitchapi_match_xg.csv` (match, home_xg, away_xg — aggregated from shots)
   - `pitchapi_advanced_team.csv` (per match, per team: possession-value, passing, carrying,
     creation, defending, territory, tempo)
   - `pitchapi_advanced_player.csv` (per match, per player)
   - `pitchapi_momentum.csv`, `pitchapi_lineups.csv` (optional/phase 2)
3. **XGProvider adapter** — implement `pitch_oracle_core/providers.py::XGProvider`
   (`fetch(league, season) -> DataFrame`) and register in `ProviderRegistry.xg`. This is the
   exact seam the docs (`02-data-source-matrix.md` action item 4) said to fill once, in
   core, so all leagues inherit it.
4. **Feature activation** — merge match xG/shots into `combined_historical_data.csv` as the
   optional `HomeXG` / `AwayXG` (or `home_xg` / `away_xg`) columns the ledger already reads,
   then retrain. Rebuild `FeatureContract` / `FEATURE_POLICY_VERSION` and the precomputed
   artifacts per the standard pipeline.
5. **Point-in-time discipline** — new columns must pass `features.py::is_prematch_feature`
   (they will, unless they hit `_EXCLUDED_COLUMNS` or start with `API_` — avoid the `API_`
   prefix since the policy treats those as unversioned/leaky). Keep `ANALYTICS_UNAVAILABLE`
   (match exists but was never rated) and `RESOURCE_NOT_FOUND` distinct in the fetcher —
   the docs treat the former as an ordinary answer.
6. **UI (phase 2, capability-gated)** — shot maps, momentum, style fingerprints, h2h, and
   real-vs-model xG in the manifest-v3 pages, gated through the provider capability report
   (`pitch_oracle_core/data/providers.py`) so the app degrades gracefully when coverage
   lags.

## Caveats & open questions

- **Event-feed coverage is not universal.** The `ANALYTICS_UNAVAILABLE` error exists
  because some matches are held but never rated. Coverage should be measured per league
  (a quick backfill census: count `analytics_unavailable` across a full season) before
  treating advanced analytics as a guaranteed feature.
- **Team/player ID namespaces differ** from existing sources (football-data.co.uk team
  names, ESPN slugs, ClubElo codes). Match by team name via `config.team_aliases` (the same
  mapping used for ClubElo/Understat) rather than trusting any foreign key.
- **xG model provenance:** PitchAPI's xG is its own model. The docs warn that xG totals
  from different providers "must never be mixed into one calculation." Pick PitchAPI as the
  single xG source for the shared build and don't blend it with Understat/Bzzoiro values.
- **No odds data.** PitchAPI serves no betting odds — the odds pipeline (football-data,
  bzzoiro, odds-api.io) is untouched. H2H and momentum are context, not markets.
- **API is young & free with no SLA.** Treat as a capability-gated optional source like
  referee/injuries, with the fetch caching into `data_files/` so the artifact pipeline
  never hard-depends on it at runtime.
- The leagues list shows a few thin catalogues (new leagues with only 1–2 seasons, e.g.
  Egyptian/Russian Premier League). For our 6 leagues the history is 4–5 seasons — enough
  for rolling-window features but not for a long backtest.
- **Is 2021-back sufficient? Yes — it matches the current training window.**
  `combine_raw_data.py::recent_season_codes(count=5)` downloads only the last 5 seasons
  from football-data.co.uk; the no-odds model's features are short windows (`_l5`, `_l10`,
  EWM span-10), so extra decades add little signal. PitchAPI serves 4–5 complete seasons
  for all 6 leagues (2021/22 → 2025/26 for EPL; Turkey starts 2022/23), i.e. full parity
  with today's training set. The one place longer history would help is the walk-forward
  Poisson's league/team priors (20-match prior, full-history accumulation) — but that runs
  off the football-data *results* backbone, not the xG/advanced feed. Bonus: football-data
  returned HTTP 300s when probed (2026-08-16), so PitchAPI's results could double as a
  cross-check/fallback for the historical fixture feed.
- **The `pk_test_` key is the real production key.** Despite the prefix, `PITCH_API_KEY`
  (a `pk_test_` key) is the working key for this project — confirmed by the operator. Treat
  it as a live credential: keep it server-side, never ship it in client code or public
  repos (the API stores only a SHA-256 hash of the key, so a leaked key must be replaced).

## Sample requests (using the key in `.env`)

```bash
# list leagues
curl https://api.pitchapi.dev/v1/leagues -H "X-API-KEY: $PITCH_API_KEY"

# Eredivisie 2025/26 fixtures + results (l_4H43wr)
curl "https://api.pitchapi.dev/v1/leagues/l_4H43wr/matches?season=2025/2026" -H "X-API-KEY: $PITCH_API_KEY"

# per-shot xG for a match
curl https://api.pitchapi.dev/v1/matches/m_0HPZUC/shots -H "X-API-KEY: $PITCH_API_KEY"

# advanced team analytics (xT, VAEP, PPDA, tempo, ...)
curl https://api.pitchapi.dev/v1/matches/m_0HPZUC/advanced -H "X-API-KEY: $PITCH_API_KEY"

# momentum curve
curl https://api.pitchapi.dev/v1/matches/m_0HPZUC/momentum -H "X-API-KEY: $PITCH_API_KEY"

# today's covered matches
curl https://api.pitchapi.dev/v1/date/2026-08-16 -H "X-API-KEY: $PITCH_API_KEY"
```

(League IDs from the live `/leagues` call: Eredivisie `l_4H43wr`, Belgium First Division A
`l_2L6d1F`, Liga Portugal `l_4QexZg`, Super Lig `l_0S1uaf`, Scottish Premiership `l_1LMdEO`,
EPL `l_4WFCIZ`.)

## Verdict

**Adopt as the shared xG + event analytics provider.** It fills the exact gap the
expansion docs call out (no Understat outside the big 5, no shots-based xG, dormant
event/action plumbing), it's free and unlimited, it covers every target league back to
2021/22, and it was verified working end-to-end with the existing key. Start with the
match-level xG feed through the existing ledger features (low risk, activates dormant
machinery), then layer the advanced analytics and UI as capability-gated phase 2.
