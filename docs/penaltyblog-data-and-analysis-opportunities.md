# penaltyblog Data and Analysis Opportunities

This is the opportunity map for incorporating penaltyblog-derived data or
analysis into Pitch Oracle and its league consumers. It separates source
availability from model usefulness: a capability can be technically available
and still be a poor production feature.

## Opportunity matrix

| Opportunity | Data required | Best consumer scope | Value | Main risk | Recommendation |
|---|---|---|---|---|---|
| Point-in-time ClubElo strength | ClubElo snapshots or history, entity aliases | All leagues with ClubElo coverage | Strong low-cost prior and drift signal | Future ratings accidentally joined to old fixtures | Ship as lagged rating artifact |
| Football-data.co.uk historical results | Results, dates, odds where present | All supported competitions | Stable goals/outcome baseline | Source revisions and delimiter/schema changes | Keep as immutable raw snapshots plus normalized table |
| FBRef team/player statistics | Possession, shots, creation, defensive actions | Consumers with stable coverage | Adds non-goal team style/context features | Scraping fragility, rate limits, role leakage | Add as optional supplemental provider after source spike |
| Understat xG | Shot-level and fixture xG | EPL/big-five style competitions only | Strong attacking/defensive signal | Not available for most target leagues; provider behavior | Capability-gate; never make the core contract require it |
| StatsBomb open event data | Events, lineups, match metadata | Research and leagues with matching coverage | xT, pressure, action-value, style features | Sparse competition coverage and event semantics | Build a research artifact first |
| Opta event/API feeds | Paid events, stats, injuries, lineups | Licensed consumers only | Broadest production event coverage | Credentials, cost, contract/licensing | Optional MatchFlow connector, never default |
| Market-implied expected goals | 1X2/totals odds, timestamps, bookmaker identity | Market-aware track | Excellent benchmark and quote-quality signal | Leakage, stale quotes, de-vig assumptions | Ship as a separate market artifact |
| Probability-grid markets | Fitted goal model or lambdas | All consumers | Removes duplicated market math | Truncation and normalization disagreement | Centralize through one adapter |
| Elo/Massey/Colley/Pi | Chronological results | All consumers | Diverse rating baselines and disagreement signal | Full-season leakage; different scales | Lagged features plus diagnostic dashboard |
| Dixon-Coles and alternatives | Historical goals, team IDs, recency weights | All consumers | Fast, interpretable distributional baseline | In-sample selection and unstable sparse leagues | Keep DC champion candidate; evaluate challengers OOF |
| Hierarchical Bayesian goals | Historical goals across teams/leagues | Sparse or cross-league research | Shrinks weak teams and quantifies uncertainty | Slow training and convergence failures | Research-only until diagnostics and runtime are proven |
| xT | Normalized event coordinates, action outcome | Event-enabled consumers | Style, progression, and player/team contribution | Provider-specific coordinate/event labels | Per-competition/season artifact with held-out validation |
| Upstream betting/backtest helpers | Historical quotes and settlement outcomes | Research/EPL market consumer | Repeatable strategy experiments | Survivorship, closing-price, and execution bias | Wrap existing ledger; do not replace it |
| FPL optimizer | FPL API, prices, fixtures, scoring rules | EPL consumer only | Product expansion and fantasy analytics | Not transferable to other leagues | Isolate in an EPL package |

## Highest-value feature families

The following feature families are promising because they are interpretable,
can be made point-in-time safe, and fit the existing feature/model contracts.

### 1. Distribution and model-disagreement features

For each fixture, retain the full score distribution and compare a small set of
approved candidates:

```text
dc_p_home, dc_p_draw, dc_p_away
poisson_p_home, poisson_p_draw, poisson_p_away
bp_p_home, bp_p_draw, bp_p_away
market_p_home, market_p_draw, market_p_away
model_disagreement_l1
model_disagreement_entropy
dc_market_residual_home
dc_market_residual_draw
dc_market_residual_away
score_grid_tail_mass
```

These should be generated from the same fixture issue timestamp. The ensemble
can learn when a prediction is fragile without treating disagreement as proof
that one model is correct.

### 2. Point-in-time strength and schedule pressure

Use ClubElo and the internal ratings history to create:

```text
elo_home_pre, elo_away_pre, elo_diff_pre
elo_delta_28d_home, elo_delta_28d_away
pi_home_pre, pi_away_pre
rating_consensus_mean, rating_consensus_std
matches_last_14d_home, matches_last_14d_away
days_since_last_match_home, days_since_last_match_away
travel_or_neutral_flag
```

All values must come from the most recent completed match before the issue time.
The rating standard deviation is especially useful as a diagnostic: high
disagreement between rating systems can be displayed as “evidence is mixed,”
not translated automatically into a higher or lower bet.

### 3. Market residual and quote quality

Create a market-aware analysis table separate from independent model features:

```text
market_overround
market_devig_method
market_quote_count
market_quote_age_seconds
market_implied_home_xg
market_implied_away_xg
model_minus_market_home_xg
model_minus_market_1x2_l1
closing_line_value
```

This supports three valuable reports:

- **Model versus market:** where the forecast differs and why.
- **Market quality:** which fixtures have thin, stale, or high-margin quotes.
- **Calibration by quote quality:** whether model performance changes when the
  market is liquid versus sparse.

These fields must never leak into the independent model track. Keep them in a
separate artifact and enforce the existing `ForecastTrack` rules.

### 4. xT and event-style features

For each competition-season model, aggregate xT into a small, stable feature
set before creating a long tail of experimental features:

```text
xt_added_per90
xt_from_passes_per90
xt_from_carries_per90
xt_conceded_per90
xt_progression_share
xt_final_third_entry_rate
xt_style_surface_distance
```

Do not mix raw xT totals from different sample sizes. Retain action counts,
minutes, source, coordinate schema, fit season, and held-out validation score.
Use xT first for team snapshots and match narratives; promote it to the goal
ensemble only after it survives a season-held-out test.

### 5. Calibration and decision-quality analysis

The most valuable “analysis” contribution is probably not another model. It is
making each market decision explainable:

```text
forecast_probability
market_probability
fair_price
offered_price
edge
expected_return
forecast_entropy
calibration_bucket
quote_freshness
model_version
source_coverage
```

Report proper scores, reliability, expected calibration error, closing-line
value, drawdown, and coverage by league, season, favorite status, goal-total
bucket, and source completeness. This turns penaltyblog's market math into
auditable product analysis rather than just additional rows in a prediction
CSV.

## Creative, testable analyses

### A. Score-grid tail stress test

Run every production fixture at `max_goals=10`, `15`, and `20`. Measure how much
the 1X2, totals, and handicap prices change. If a fixture has material tail mass,
do not normalize it silently; mark the market surface as low confidence or use a
higher grid.

### B. Market-derived goal residual map

Invert 1X2 plus Over/Under 2.5 into market-implied goal rates, then chart:

```text
model_home_xg - market_home_xg
model_away_xg - market_away_xg
```

Slice by bookmaker, league, kickoff lead time, and quote coverage. This can
reveal whether a model's apparent edge is really a consistent market bias, a
stale quote, or a team-data problem.

### C. Promotion/relegation prior transfer

For sparse leagues, compare three starts for newly promoted teams:

1. league-average prior;
2. ClubElo prior;
3. hierarchical Bayesian prior shared across competitions.

Use only post-promotion fixtures for evaluation and report calibration during
the first 5, 10, and 20 matches. This is a much safer experiment than using
cross-league information indiscriminately.

### D. “Evidence disagreement” narrative

Generate a transparent narrative when goal model, ratings, xT, and market
baseline disagree:

```text
Goal model: home edge
Ratings: home edge
xT: away progression edge
Market: near-even
Conclusion: mixed evidence; suppress high-confidence staking output
```

This is useful product intelligence even when it does not improve a score. It
helps users understand uncertainty without inventing causal explanations.

### E. Provider ablation report

For each consumer, run the same rolling-origin folds with:

- goals only;
- goals + ClubElo;
- goals + ratings;
- goals + xT;
- goals + market-aware features;
- full available feature set.

Report score gain, coverage loss, freshness failures, and maintenance cost. A
feature source should earn its place by improving forecast or decision quality,
not because it is technically interesting.

## Source and licensing/operations policy

Every upstream-backed artifact should record:

```text
provider
source_url_or_endpoint
retrieved_at
source_revision_or_snapshot_hash
coverage_start
coverage_end
entity_registry_version
provider_terms_reviewed_at
```

Scrapers should run in scheduled ingestion jobs with caching, bounded retries,
and a declared minimum request interval. Streamlit should read the resulting
artifact and render an honest unavailable/stale state when the source is not
present. Paid Opta or private StatsBomb connections should be isolated behind
consumer capability flags and must not change the core package's import-time
requirements.

## Suggested experiment backlog

| Experiment | Primary metric | Pass condition |
|---|---|---|
| Dixon-Coles versus current champion | Rolling-origin log loss and RPS | Better primary score with no material calibration regression |
| Bivariate Poisson challenger | Log loss, scoreline ignorance | Improvement on leagues with residual goal covariance |
| Rating features | Brier/log loss by season | Stable gain in at least two held-out seasons |
| xT features | Team/fixture OOF score and coverage | Gain after controlling for event coverage and sample size |
| Market-implied xG comparator | CLV and calibration | Better market diagnostics without independent-track leakage |
| Hierarchical Bayes | Early-season calibration | Improvement in sparse leagues with converged diagnostics |
| Tail stress | Market probability delta | No production surface with unbounded or material truncation error |
| MatchFlow versus eager parsing | Runtime, memory, reproducibility | Lower peak memory with identical golden outputs |

## Recommendation by product tier

### Shared core

Keep: goal-model adapters, probability-grid conversion, implied odds,
provider-neutral metrics, ratings primitives, artifact provenance, and strict
version tests.

### League consumers

Add only: source adapters with confirmed coverage, league-specific mappings,
point-in-time feature joins, xT/event artifacts where available, and consumer
capability flags.

### Research/EPL consumer

Experiment with: Understat, FPL, hierarchical Bayes, upstream backtesting,
Opta/StatsBomb private feeds, and richer `viz.Pitch` surfaces.

This division gives Pitch Oracle the analytical leverage of penaltyblog without
turning every league repository into a copy of the upstream project's entire
dependency and data model.

