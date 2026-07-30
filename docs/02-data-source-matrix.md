# Data Source Matrix: premier-league's 8 Sources × 5 New Leagues

`premier-league` integrates 8 data sources. This is the same coverage check from our
earlier conversation, now mapped explicitly against what the app actually uses.

| Source | Used for | Scotland | Eredivisie | Portugal | Belgium | Turkey |
|---|---|---|---|---|---|---|
| **football-data.co.uk** (`combine_raw_data.py`, `prepare_model_data.py`) | Core historical results + odds, 1993–95 onward | ✅ full | ✅ full | ✅ full | ✅ full | ✅ full, thinner odds |
| **ESPN API** (`fetch_upcoming_fixtures.py`) | Upcoming fixtures, kickoff times | ✅ `sco.1` | ✅ `ned.1` | ✅ likely `por.1` | ✅ `bel.1` | ⚠️ verify slug |
| **ClubElo** (`fetch_clubelo.py`) | Team strength ratings | ✅ | ✅ confirmed `NED_1` | ✅ | ✅ | ✅ |
| **API-Football** (`fetch_api_football.py`) | Supplemental fixtures/stats, paid tier | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Understat xG** (`fetch_understat_xg.py`) | Expected-goals features | ❌ not covered | ❌ not covered | ❌ not covered | ❌ not covered | ❌ not covered |
| **Weather** (`fetch_weather_data.py`) | Match-day conditions feature | ✅ needs new stadium coords | ✅ needs new stadium coords | ✅ needs new stadium coords | ✅ needs new stadium coords | ✅ needs new stadium coords |
| **Referee scraping** (`scrape_referees.py`, `analyze_referee_impact.py`) | Referee tendency features | ⚠️ verify source | ⚠️ verify source | ⚠️ verify source | ⚠️ verify source | ⚠️ verify source |
| **Injury scraping** (`scrape_injuries.py`, `scrape_injuries_web.py`) | Injury/availability features | ⚠️ verify source | ⚠️ verify source | ⚠️ verify source | ⚠️ verify source | ⚠️ verify source |
| **Bzzoiro odds API** (`bzzoiro_football_api.py`) | Live market odds shown alongside model prediction | ⚠️ verify | ⚠️ verify | ⚠️ verify | ⚠️ verify | ⚠️ verify |

## What this means for feature parity

The two features that quietly do the most work in `premier-league`'s accuracy gains —
**Understat xG** and (to a lesser extent) the **referee/injury scrapers** — don't
transfer cleanly:

- **Understat is a hard no across all 5 leagues.** Understat only covers the "big 5" +
  RFPL. Any xG feature engineering currently keyed off Understat needs either (a) an
  xG proxy computed from shot-count/location data you don't have for these leagues,
  or (b) dropped entirely for the new repos, with the model retrained/reweighted
  without it. Don't try to fake it — a missing-data imputation for xG across 5 whole
  leagues will just inject noise into the ensemble.
- **Referee and injury scrapers are unverified and are the first thing to spike.**
  `scrape_referees.py` targets Playmaker Stats in the current repo; coverage for
  Scotland/Netherlands/Portugal/Belgium/Turkey needs to be checked page-by-page
  before you assume the scraper "just works" with a URL swap — site structure and
  data depth often differ by country on these aggregators. Same caution for
  `scrape_injuries_web.py`. Budget a scoping day per source, per league, before
  committing to including these as launch features.
- **Bzzoiro** (the odds source shown next to predictions in the app) needs its
  documented coverage checked directly — don't assume parity.

## What transfers cleanly

- **football-data.co.uk** — full parity, ~30 years of results + odds, same CSV shape,
  same `Div` code and column layout you already parse for EPL (`E0`). This is the
  backbone data source and it's not a weak link for any of the 5.
- **ClubElo** — genuinely global coverage; the `NED_1`-style country/level identifier
  pattern (confirmed via the `soccerdata` package's built-in league table) should
  extend cleanly to `SCO_1`, `POR_1`, `BEL_1`, `TUR_1` with the same fetch logic,
  just a different code.
- **ESPN fixtures** — Scotland and Belgium both have live ESPN fixture/standings
  pages under `sco.1` and `bel.1` respectively (confirmed by direct page fetch).
  Netherlands is confirmed via the `soccerdata` mapping (`ned.1`). Portugal and
  Turkey are very likely present given ESPN's broad soccer coverage, but confirm
  the exact slug before wiring `fetch_upcoming_fixtures.py` in Phase 1 for those two.
- **API-Football** — paid tier, broad league coverage; this is your fallback/
  cross-check source regardless of the free-source gaps above.
- **Weather** — no coverage gap, just data-entry work: each new league needs its own
  stadium name → GPS coordinate lookup table built once and reused (same pattern as
  whatever `fetch_weather_data.py` already does for the 20 EPL grounds).

## Action items before Phase 1 starts

1. Spike-test `scrape_referees.py` and `scrape_injuries_web.py` against one league
   (recommend Eredivisie, since it's structurally simplest — see
   `04-per-league-notes.md`) to decide whether referee/injury features ship at
   launch or land in a v2.
2. Confirm the ESPN API slugs for Portugal and Turkey.
3. Confirm what Bzzoiro actually covers.
4. Decide the Understat replacement strategy (drop the feature vs. build a
   shots-based xG proxy) once, in `pitch-oracle-core`, so all 5 leagues inherit the
   same answer instead of 5 separate ad-hoc decisions.
