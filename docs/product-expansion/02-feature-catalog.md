# Feature catalog: 50 universal capabilities

Each feature has a stable ID used by the implementation map and roadmap. “Universal”
means the feature contract works for every league; optional inputs produce explicit
fallback states rather than league-specific forks.

Priority meanings:

- **P0** — correctness/trust prerequisite.
- **P1** — first major product release.
- **P2** — high-value analysis after the match experience is sound.
- **P3** — advanced modeling or optional-provider intelligence.

## Match intelligence

| ID | Feature | User experience | Engine and fallback | Priority / done when |
|---|---|---|---|---|
| F01 | Match Intelligence Center | One page per fixture with kickoff, venue, forecast, goal outlook, form, availability, weather, and season stakes. | Joins `forecast`, `team_snapshot`, `fixture_context`, and capability flags by canonical `fixture_id`. Missing optional blocks explain why. | P1 — every scheduled fixture has a stable deep link and no page joins on display names. |
| F02 | Scoreline probability matrix | Interactive 0–0 through 6–6 heatmap with tail probability and highlighted modal scores. | Dixon-Coles/bivariate score distribution; current independent Poisson is the fallback. | P1 — visible cell mass plus tail equals 1 within tolerance. |
| F03 | Full goal-market ladder | Probabilities for O/U 0.5–5.5, BTTS, clean sheets, win-to-nil, double chance, draw-no-bet, and team totals. | Derive all coherent markets from one score matrix. | P1 — mechanically reconcile every market to the same joint distribution. |
| F04 | Forecast driver waterfall | “Why this forecast?” shows the strongest home/away/draw drivers and their direction. | Precomputed SHAP for tree models; standardized coefficient contributions or leave-one-feature-out deltas for other models. | P1 — explanation identifies model/version and never uses post-match data. |
| F05 | Evidence-backed narrative | Three concise bullets: base strength, current form/context, and uncertainty caveat. | Deterministic templates over typed evidence; no free-form unsupported claims. | P1 — every claim links to a visible metric and timestamp. |
| F06 | What-if scenario lab | Toggle lineup loss, extra rest, home advantage, weather, and tactical pace; compare forecast deltas. | Recompute only approved mutable features through the same inference pipeline. Unsupported controls are disabled. | P2 — reset reproduces the cached forecast within `1e-6`. |
| F07 | Forecast uncertainty fan | Show median probability plus 50%/80% intervals and “stable vs fragile” classification. | Bootstrap model/parameter draws and conformal residual bands; sample-size bands as fallback. | P1 — backtest reports empirical interval coverage. |
| F08 | Upset Radar | Rank fixtures where the weaker team has unusually high win probability. | Difference between forecast and dynamic-strength baseline, filtered by uncertainty. | P1 — promoted/cold-start teams are labeled rather than overstated. |
| F09 | Draw Radar | Rank draw-prone fixtures using draw probability, low scoring, team parity, and draw calibration. | Calibrated draw head plus score-matrix draw mass. | P1 — draw reliability is published separately. |
| F10 | Goal Fest / Low Block index | Two watchlists for high-event and low-event fixtures. | Expected total goals, score entropy, BTTS, shots tempo, and style matchup. | P1 — threshold is league-percentile based, not a universal raw cutoff. |
| F11 | Match context timeline | Chronological form, manager changes, rest, travel, and forecast revisions before kickoff. | Append-only fixture snapshots keyed by `observed_at`. | P2 — users can distinguish initial, 24-hour, lineup, and closing forecasts. |
| F12 | Shareable forecast card | Downloadable compact PNG/HTML card with teams, time, probabilities, score mode, uncertainty, and model timestamp. | Server-side HTML component or Matplotlib template from the same forecast row. | P2 — includes data/model timestamp and responsible-use footer. |

## Team intelligence

| ID | Feature | User experience | Engine and fallback | Priority / done when |
|---|---|---|---|---|
| F13 | Team Command Center | Current rank, power rating, form, attack/defense, schedule, projections, squad status, and recent matches. | `team_snapshot` analytical mart; optional cards capability-gated. | P1 — a current team can be opened from any table or fixture. |
| F14 | Form fingerprint | Rolling points, xG/shot proxy, goals, shot quality, clean sheets, and finishing vs expectation over selectable windows. | Perspective-normalized team events with exponentially weighted windows. | P1 — home and away matches both update the same team sequence. |
| F15 | Home/away split explorer | Compare performance by venue without confusing venue-only form with overall form. | Common team ledger grouped by venue after state calculation. | P1 — labels expose sample sizes and shrink small samples to league average. |
| F16 | Opponent-adjusted performance | Shows whether recent results came against strong or weak opposition. | Residual performance relative to opponent Elo/expected goals. | P2 — no current match outcome enters its own adjustment. |
| F17 | Team comparison studio | Compare any two clubs on strength, attack, defense, tempo, discipline, rest, projections, and head-to-head. | Percentile-normalized team snapshot and common metric dictionary. | P1 — comparison works even if clubs have not met recently. |
| F18 | Head-to-head context | Recency-weighted meetings, venue context, scorelines, and a sample-quality warning. | Canonical IDs and exponential time decay; “insufficient/relevance low” fallback. | P2 — role-reversed historical meetings count correctly. |
| F19 | Dynamic power rankings | League leaderboard with Elo/Glicko strength, attack/defense components, and week-over-week movement. | Online rating updated after each completed fixture. | P1 — replaying identical history is deterministic. |
| F20 | Style fingerprints and clusters | Labels such as high press proxy, direct/high-event, possession-control proxy, or low block based on available event aggregates. | League-relative clustering with documented input coverage; use neutral statistical labels when rich events are absent. | P3 — cluster stability and feature definitions are published. |
| F21 | Fixture difficulty calendar | Color-coded past and future schedule with opponent strength, venue, rest, and expected points. | Forecast probabilities plus dynamic opponent strength. | P2 — difficulty is evaluated as of each fixture, not with final-season ratings. |
| F22 | Congestion, travel, and recovery load | Flags short rest, long travel, timezone change, and clusters of matches. | UTC kickoff, stadium coordinates, and all-competition fixture feed; rest-only fallback. | P2 — travel is omitted when venue confidence is low. |
| F23 | Manager-change tracker | Before/after performance, shrinkage-aware “bounce,” tenure, and forecast impact. | Point-in-time manager tenures; hidden when provider absent. | P3 — manager assignment is valid at kickoff, never current-manager backfill. |
| F24 | Squad availability impact | Missing player minutes/quality, likely replacements, and scenario delta. | Point-in-time availability plus player strength; count-only fallback if quality unavailable. | P3 — stale reports show age and source. |
| F25 | Discipline and referee matchup | Team card tendencies against referee foul/card/penalty profile. | Point-in-time referee assignment/history; team discipline alone when referee unavailable. | P3 — minimum referee sample and shrinkage are displayed. |

## League and season intelligence

| ID | Feature | User experience | Engine and fallback | Priority / done when |
|---|---|---|---|---|
| F26 | Rule-aware live table | Correct regular/split/playoff standings, sanctions, points transitions, tie-breakers, and games in hand. | Versioned `CompetitionRules`; no generic silent fallback for unsupported rules. | P0 — reproduces an independently verified table. |
| F27 | Season projection table | Expected final points/rank and probability of every finishing position. | Monte Carlo remaining-fixture simulations over score distributions. | P1 — probabilities per team and per position sum correctly. |
| F28 | Title/Europe/relegation race | Named outcome probabilities and threshold bands appropriate to that competition edition. | Rules map ranks/phases to outcomes; generic rank distribution remains if labels unavailable. | P1 — labels derive from edition rules, not league-name conditionals. |
| F29 | Matchday stakes index | Quantifies how much a fixture can change qualification/relegation outcomes. | Difference in simulated outcome entropy/probabilities conditional on H/D/A. | P2 — calculation uses common random numbers for stable comparisons. |
| F30 | Points target calculator | “How many points likely needed for top four/safety?” distribution and team-specific required pace. | Simulation-derived threshold distribution. | P2 — exposes percentile range, not a false single cutoff. |
| F31 | Split/playoff scenario explorer | Preview pools, points halving, bracket paths, and qualification outcomes. | Generic rule engine consumes phases, pools, transitions, and bracket definitions. | P2 — Belgium and Scotland require no UI branch. |
| F32 | Matchday storylines | Automatically surfaces biggest rating move, upset, form swing, high-stakes fixture, and model surprise. | Deterministic ranked events over matchday artifacts. | P2 — every storyline has a supporting metric and link. |
| F33 | League trend laboratory | Time series for goals, home advantage, draws, cards, tempo, market error, and competitive balance. | Rolling competition aggregates with season/rules markers. | P2 — windows and sample sizes are visible. |
| F34 | Competitive balance dashboard | Strength dispersion, title concentration, parity index, and promotion/relegation churn. | Elo dispersion, Herfindahl/Gini-like measures, and simulation concentration. | P3 — metrics are normalized for league size. |
| F35 | Cross-league comparison | Compare leagues on tempo, scoring, home advantage, parity, forecast difficulty, and calibration. | Shared metric schema across consumer artifacts; can be aggregated by a separate index app. | P3 — comparisons use aligned seasons and definitions. |

## Forecasting and model governance

| ID | Feature | User experience | Engine and fallback | Priority / done when |
|---|---|---|---|---|
| F36 | Champion/challenger Model Lab | See deployed model, challengers, promotion reason, evaluation window, and deltas vs baselines. | Typed model registry and contextual release gate. | P1 — production choice is reproducible from artifacts. |
| F37 | Reliability and calibration explorer | Reliability curves for 1X2 and goal markets, confidence histogram, ECE/Brier by season and cohort. | Out-of-fold predictions only. | P0/P1 — no training-fit probabilities enter the chart. |
| F38 | Historical prediction tracker | Search every issued forecast, result, probability score, revision, and closing-line comparison. | Append-only prediction ledger keyed by forecast issuance time. | P1 — forecast is immutable after kickoff. |
| F39 | Dynamic Dixon-Coles goals model | Better low-score correlation and time decay than independent full-history Poisson. | Fit attack/defense/home/rho with recency weights; independent Poisson fallback on fit failure. | P1 — rolling-origin log loss/Brier gate and finite score mass. |
| F40 | Dynamic Elo/Glicko strength | Fast, interpretable team strength and promoted-team prior. | Online updates with league/country priors and season regression. | P1 — time-travel queries reproduce the rating at kickoff. |
| F41 | Calibrated ensemble | Blend goals, Elo, and feature model based on out-of-fold log-loss weights, then calibrate. | Use the best individually gated model if stacking data is insufficient. | P2 — ensemble must beat components with paired uncertainty or remain challenger. |
| F42 | Promoted/new-club transfer prior | Borrows strength from previous division, country coefficient, squad value, or conservative league prior. | Hierarchical prior with explicit provenance; league prior if no evidence. | P2 — cold-start badge remains until effective sample threshold. |
| F43 | Cross-market probability reconciliation | 1X2, scoreline, totals, BTTS, and team totals never contradict one another. | Derive from/reconcile to a joint score distribution. | P0/P1 — invariant tests pass for every fixture. |
| F44 | Forecast drift monitor | Alerts when features, team coverage, probability distribution, calibration, or residuals shift. | PSI/KS/simple z-score reports with season-aware baselines. | P1 — drift artifact has severity, evidence, and suggested action. |
| F45 | Cohort performance slices | Performance for promoted teams, early season, favorites, derbies, short rest, split phase, and data-quality bands. | Evaluation tags computed point-in-time. | P1 — minimum sample and confidence intervals prevent noisy rankings. |

## Market, context, and trust features

| ID | Feature | User experience | Engine and fallback | Priority / done when |
|---|---|---|---|---|
| F46 | Entity coverage and cold-start badges | A fixture visibly says “full history,” “partial alias match,” “promoted prior,” or “league prior.” | Entity resolution report plus effective sample sizes embedded in forecasts. | P0 — unresolved active teams fail CI unless explicitly approved as new. |
| F47 | Data freshness and provenance panel | Source, observed time, coverage, status, and last successful update for each block. | Provider-run ledger and artifact dependency graph. | P0 — every user-visible datum is traceable to an artifact/source timestamp. |
| F48 | Data quality control room | Duplicate fixtures, missing fields, stale sources, impossible scores/odds, alias gaps, and leakage checks. | Machine-readable validation results and severity gates. | P0 — critical defects block manifest generation. |
| F49 | Fair odds and market movement | De-vigged consensus, model fair price, edge, opener-to-current move, dispersion, and stale-market flag. | Timestamped odds snapshots; page hides value language when prices are absent. | P3 — all prices carry bookmaker and observation time. |
| F50 | Responsible portfolio/backtest lab | Fractional-Kelly sizing with caps, correlated exposure, drawdown simulation, and historical calibration—not a “tip” list. | Forecast ledger plus executable market prices; education-only fallback without odds. | P3 — defaults to zero stake if uncertainty, freshness, or edge gates fail. |

## High-leverage feature bundles

The features should ship in coherent bundles, not one widget at a time.

### Bundle A — Trustworthy Match Center

F01–F05, F07, F37, F43, F46–F48. This fixes the Belgium identity failure, creates a
joint probability product, and makes every forecast explainable and auditable.

### Bundle B — Season Command Center

F13–F21 and F26–F33. One team ledger and one rule engine power most of this bundle;
the marginal cost of each page becomes small after those foundations exist.

### Bundle C — Forecast Foundry

F36 and F39–F45. The output is not “more models.” It is a reliable promotion process
where a simple model remains deployed when complexity does not earn its place.

### Bundle D — Context and markets

F06, F11, F22–F25, and F49–F50. These depend on timestamped optional providers and
should remain capability-gated. They should not delay the no-odds core product.

## Product rules that apply to every feature

1. No display string is a key.
2. No historical feature can observe at or after its fixture kickoff.
3. Every number has a timestamp, definition, sample size, and source lineage.
4. League rules are data, versioned by competition edition.
5. Forecasts are distributions, not categorical “picks.”
6. Uncertainty and cold starts are visible at the point of use.
7. Optional-provider absence degrades a card, never the whole page.
8. Expensive computations run in the artifact workflow, not on Streamlit reruns.
9. A new model is a challenger until it wins an out-of-time probability gate.
10. Market value is never claimed without fresh executable prices.

