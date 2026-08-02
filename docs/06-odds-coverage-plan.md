# Odds Coverage Plan for 5 New Leagues

## What's Already Covered (No Work Needed)

**football-data.co.uk provides historical odds for all 5 leagues** — Scotland (SC0),
Netherlands (N1), Portugal (P1), Belgium (B1), and Turkey (T1) — going back to
1993/94. This is confirmed by directly inspecting the football-data.co.uk download
page: all five country flags appear in every season's CSV `data.zip` and Excel
`all-euro-data` files. The existing pipeline in `combine_raw_data.py` →
`prepare_model_data.py` (including `extract_betting_features()`) works for all five
leagues by simply changing the `Div` code. **No new odds source is needed for model
training.**

The one real concern from the docs — "thinner odds-column coverage for Turkey" —
is a feature engineering issue (fewer bookmaker columns to derive features from),
not a source availability problem. The existing column-rename map in
`prepare_model_data.py` already handles missing columns gracefully via `if col in
df.columns` guards in `extract_betting_features()`. Note from football-data.co.uk's
`notes.txt`: odds columns vary by season and league — not all bookmakers appear in
every CSV. The pipeline already degrades gracefully.

A spot-check of the N1 (Eredivisie 2024/25) CSV confirmed the standard odds layout
is identical to E0: B365, BW, BF, PS, WH, 1XB, BFE (H/D/A plus over/under, Asian
handicap, and both opening and closing variants for all bookmakers).

## The Actual Gap: Live/Upcoming Odds for App Display

The current EPL app uses **Bzzoiro API** (`bzzoiro_football_api.py`) to show live
odds (H/D/A, O2.5, BTTS) next to model predictions in the Streamlit app. Bzzoiro
only covers the Premier League (hardcoded `api_id=17`). The 5 new leagues need an
equivalent live-odds display source.

## API Options Evaluated

Three free-tier odds APIs were assessed against the requirements: (1) must cover
Scotland/NL/Portugal/Belgium/Turkey, (2) must have a genuinely free tier, (3) must
return H/D/A (1X2) odds in a machine-readable format.

### Odds-API.io ✅ Recommended

**Website:** https://odds-api.io | **Free tier:** 100 req/hr (500/day), no credit card

| Criterion | Assessment |
|---|---|
| **All 5 leagues** | ✅ Eredivisie, Scottish Premiership, Liga Portugal explicitly listed on landing page; 12,000+ total leagues cover Belgium and Turkey |
| **API shape** | REST, JSON, key-as-query-param auth |
| **League filtering** | `GET /v3/events?league={slug}` — one call per league returns all upcoming matches with event IDs and team names |
| **Odds endpoint** | `GET /v3/odds?eventId={id}&bookmakers=Bet365` — per-event, returns H/D/A prices |
| **Bookmakers** | 2 recreational bookmakers on free tier (Bet365, Unibet, etc.) — sufficient for display-only |
| **Rate limit** | 500 req/day (100/hr) — 5 leagues × ~10 matches × 1 odds call each ≈ 50 req/weekend, well within limits |
| **League slug pattern** | `{country}-{league-name}` (confirmed: docs show `england-premier-league`) |

Verified league slugs (via `GET /v3/leagues?sport=football` call — run once to
confirm):

| League | Likely slug (verify) |
|---|---|
| Scotland — Premiership | `scotland-premiership` |
| Netherlands — Eredivisie | `netherlands-eredivisie` |
| Portugal — Primeira Liga | `portugal-primeira-liga` |
| Belgium — Pro League | `belgium-first-division-a` or `belgium-jupiler-pro-league` |
| Turkey — Süper Lig | `turkey-super-lig` |

### The Odds API ❌ Not recommended

**Website:** https://the-odds-api.com | **Free tier:** 500 credits/month

Fully covers all 5 leagues with confirmed sport keys (`soccer_spl`,
`soccer_netherlands_eredivisie`, `soccer_portugal_primeira_liga`,
`soccer_belgium_first_div`, `soccer_turkey_super_league`). Good API, single-call
design (one request returns all odds for a league). **Rejected solely on rate
limits:** 500 requests/month is ~12 calls/week across 5 leagues — too tight for a
weekly-updating app with odds display per match. For comparison, odds-api.io gives
500 requests **per day** at the same price (free).

### TheRundown.io ❌ Not recommended

**Website:** https://therundown.io/api | **Free tier:** 20,000 data points/day

Excellent API for US sports (NFL, NBA, MLB, NHL, NCAA) with sub-second WebSocket
updates. Soccer coverage is limited to: EPL, La Liga, Bundesliga, Serie A, Ligue 1,
MLS, Champions League, J-League, and UEFA internationals. **None of the five target
leagues are covered.** Free tier also has a 5-minute data delay and only 3
US-focused sportsbooks (BetMGM, DraftKings, FanDuel). Wrong fit for European
domestic leagues.

## Implementation Approach

The codebase already has the scaffolding for this. `pitch_oracle_core/odds.py`
defines:

- `OddsMarket` — outcome name + decimal price + bookmaker
- `OddsEvent` — event_id, home_team, away_team, start_time, markets list, provider
- `OddsAdapter` Protocol — `name` + `fetch(league, date)` → `list[OddsEvent]`
- `normalize_odds()` — converts a dict payload into an `OddsEvent`

And `pitch_oracle_core/sources.py` tracks `SourceAvailability.live_odds` and
provides `OptionalFeatureSet` for per-league optional feature gating.

### What to build

1. **`pitch_oracle_core/odds_oddsapiio.py`** — New adapter implementing `OddsAdapter`
   - Reads `ODDSAPI_IO_KEY` from `.env`
   - `fetch(league, date)` makes two calls:
     a. `GET /v3/events?sport=football&league={slug}&from={date}&apiKey={key}` — fetches upcoming matches with team names and event IDs
     b. `GET /v3/odds?eventId={id}&bookmakers=Bet365&apiKey={key}` — for each event, fetches H/D/A prices
   - Normalizes each response into `OddsEvent` using the existing `normalize_odds()` helper
   - Caches results for 1 hour (avoids redundant calls across league fetches within a session)
   - Returns `[]` gracefully if API is down, key is missing, or league slug is wrong.

2. **Per-league config** — Each league's thin config gets:
   - `odds_provider = "oddsapiio"`
   - `odds_league_slug = "netherlands-eredivisie"` (confirmed via one-time `/v3/leagues` call at setup)
   - `odds_bookmakers = ["Bet365"]` (single bookmaker on free tier; can add a second later)

3. **App integration** — The league's Streamlit entrypoint calls
   `adapter.fetch(league=config.odds_league_slug)` and displays odds in the
   same league-neutral UI pattern used by `pitch_oracle_core.ui_pages`.

### Files to touch

| File | Change |
|---|---|
| `pitch_oracle_core/odds_oddsapiio.py` | **NEW** — Odds-API.io adapter (~100 lines) |
| `pitch_oracle_core/__init__.py` | Export `OddsApiIoAdapter` |
| `pitch_oracle_core/config.py` | Add `odds_league_slug` and `odds_bookmakers` fields |
| Per-league `config.py` (5 files) | Set league-specific slugs |
| `pitch_oracle_core/ui_pages.py` | Wire adapter into the league-neutral predictions page |
| `docs/02-data-source-matrix.md` | Remove Bzzoiro row; add Odds-API.io row with coverage notes |

### What to skip

- **Live odds as model features** — the existing `extract_betting_features()` uses
  historical pre-match odds from football-data.co.uk for training. Live/upcoming
  odds from the display API are completely separate. Don't merge them.
- **Odds-API.io for historicals** — football-data.co.uk already provides 30 years of
  odds across 10+ bookmakers. No need to supplement.
- **Additional bookmakers on free tier** — 2 bookmakers is enough for "here's what
  the market thinks" display. The model doesn't use these numbers.
- **O2.5 and BTTS markets from the API** — fetch only H/D/A (1X2) odds initially.
  Add other markets later if the UI calls for it. Keeps the adapter simple.

## Verification

1. Get a free API key from https://odds-api.io (no credit card)
2. Call `GET https://api.odds-api.io/v3/leagues?sport=football&apiKey=YOUR_KEY`
   — confirm all 5 league slugs and save them to the per-league configs
3. For each league, call `GET /v3/events?sport=football&league={slug}&apiKey=YOUR_KEY`
   — verify upcoming matches return with team names and event IDs
4. Pick one event ID per league and call
   `GET /v3/odds?eventId={id}&bookmakers=Bet365&apiKey=YOUR_KEY`
   — verify response shape matches expectations (H/D/A prices in the `bookmakers` object)
5. Run `combine_raw_data.py` with `Div='N1'` (Eredivisie) — confirm odds columns
   flow through `prepare_model_data.py` without errors
6. Spot-check a Turkey CSV (T1, any season) to quantify which bookmaker columns
   are present vs. missing compared to EPL, and verify `extract_betting_features()`
   column-availability gates work correctly with the thinner column set
