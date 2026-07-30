# Per-League Structural Notes

Config values reference the identifiers used by football-data.co.uk (`Div`), ESPN,
and ClubElo where confirmed.

## Scotland — Scottish Premiership
- `Div=SC0` (football-data.co.uk), ESPN `sco.1`
- Season starts Fri Jul 31, 2026
- 12 teams, each plays every opponent **3 times** (33 games), then the table splits
  into top-6 / bottom-6 for a final round of matches against only teams in their
  half
- **Schema requirement:** a `phase` field (`regular` / `split`) and split-aware
  standings logic — points and goal difference carry over from the regular season
  into the split, but a team can no longer play (or be predicted against) teams in
  the other half
- Also has a Premiership promotion/relegation playoff against the Championship
  runner-up, outside the core season — decide whether that's in scope or explicitly
  excluded from the model's remit

## Netherlands — Eredivisie
- `Div=N1`, ESPN `ned.1`, ClubElo `NED_1` (confirmed)
- Season starts Fri Aug 7, 2026
- Straight double round-robin, 18 teams, no split or playoff complexity in the
  regular title race (there is a separate European-qualification playoff among
  mid-table teams post-season, similar caveat to Scotland's promotion playoff —
  scope decision, not a blocker)
- Structurally the simplest of the five — recommended pilot league

## Portugal — Primeira Liga
- `Div=P1`, ESPN slug unconfirmed (verify before Phase 3)
- Season starts weekend of Aug 7–10, 2026
- 18 teams, straight double round-robin, no split format
- Historically dominated by three clubs (Benfica, Porto, Sporting) — not a schema
  issue, but worth knowing if you're tuning a "competitive balance" feature that
  assumes EPL-style parity; base rates will look very different

## Belgium — Belgian Pro League
- `Div=B1`, ESPN `bel.1` (confirmed)
- Season starts weekend of Aug 7–9, 2026
- Split-season format: regular season (16 teams, 2026–27 expands to **18 teams** —
  flag this as a mid-build schema change, not a one-time config value) → points
  halved → Champions' Play-offs / Europe Play-offs / Relegation Play-offs among
  sub-groups of the table
- **Schema requirement:** same `phase` field pattern as Scotland, but with three
  named playoff pools instead of a binary split, and a points-halving
  transformation applied at the phase boundary — this is a different shape from
  Scotland's split even though both need a `phase` field, so don't assume one
  implementation covers both without checking

## Turkey — Süper Lig
- `Div=T1`, ESPN slug unconfirmed (verify before Phase 4)
- Season starts weekend of Aug 14–17, 2026 — your latest start date, which also
  gives you the most runway
- 20 teams (2025–26 onward), straight round-robin, no split/playoff structure
- Historical points deductions from match-fixing sanctions create table anomalies in
  past seasons (most notably 2011 and subsequent years) — a `points_adjustment`
  field lets you preserve the true underlying performance data for model training
  while still reflecting official final standings elsewhere in the app
- Thinnest odds-column coverage on football-data.co.uk of the five — expect fewer
  bookmakers per match in the historical CSVs, which affects any feature that
  averages/medians across books
