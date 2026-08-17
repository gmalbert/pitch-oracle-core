# Research review: football forecasting and betting markets

This chapter converts the academic literature into engineering decisions for Pitch
Oracle. It is not a claim that a historically profitable paper remains profitable,
that one league's result generalizes to every European competition, or that a more
complex model will beat the market after data, timing, limits, and execution costs.

The review prioritizes original papers, author/university copies, publisher records,
and first-party research repositories. Findings are divided into four dispositions:

- **Adopt** — well-supported method or evaluation principle that improves the platform
  without requiring unavailable data.
- **Experiment** — credible challenger that must pass Pitch Oracle's rolling-origin
  gates.
- **Capability-gated** — useful only where a documented provider supplies the required
  point-in-time data.
- **Reject/defer** — weak transferability, untestable operational assumptions, or a
  result that would make the product less auditable.

## Executive research conclusions

1. Independent Poisson is a baseline, not a default truth. Low-score dependence,
   score correlation, dispersion, and changing team strength deserve explicit tests.
2. Dynamic, recency-weighted team strength is consistently more defensible than
   full-history static form. Hierarchical partial pooling is especially relevant to
   promoted and sparsely observed clubs.
3. A complete joint score distribution is more valuable than a collection of unrelated
   classifiers because it supports internally coherent 1X2, scoreline, totals, BTTS,
   handicap, and season simulation outputs.
4. Football probabilities must be evaluated with proper scores, calibration,
   discrimination/sharpness, and cohort stability. Accuracy and ROI are insufficient.
5. Bookmaker prices are a strong forecast baseline. Different de-vig methods and source
   books can matter; market probabilities must be timestamped and kept distinct from
   the independent champion.
6. Positive historical betting returns can arise without superior forecast accuracy.
   Many-model and many-threshold searches need family-wise data-snooping controls and a
   genuinely untouched forward test.
7. Event-derived xG, xT, VAEP, player ratings, and tracking models can create genuinely
   new analysis, but they are not universal inputs until provider coverage is universal.
8. Full Kelly is inappropriate when probabilities are estimated with error. The product
   should default to zero exposure under stale data, poor calibration, or uncertain edge,
   and cap any research sizing policy aggressively.

## Evidence scale

Every proposal later in this chapter receives an evidence/transfer score:

| Level | Meaning |
|---|---|
| E4 | Multiple relevant primary studies or a foundational statistical result; direct fit to Pitch Oracle. |
| E3 | A strong football-specific study with out-of-sample evidence, but limited leagues/periods. |
| E2 | Plausible single study, research code, or evidence from another sport; experiment only. |
| E1 | Illustrative idea without adequate independent, temporal, or operational validation. |

Evidence level controls the priority of an experiment, never automatic promotion.

## 1. Score-generating models

### 1.1 Dixon-Coles: keep it as the first serious challenger

[Dixon and Coles (1997)](https://doi.org/10.1111/1467-9876.00065) model team attack
and defense through Poisson intensities, adjust low scorelines, weight past matches,
and connect forecasts to betting-market analysis. This supports F39 and the existing
implementation plan.

Disposition: **Adopt as a registered P1 challenger (E4)**.

Required refinements:

- tune decay only inside rolling training windows;
- retain a no-decay and independent-Poisson baseline;
- report the fitted low-score parameter and its stability;
- reject fits with invalid low-score mass or material truncated tail;
- never treat one reported betting result as the promotion criterion.

### 1.2 Bivariate and diagonal-inflated Poisson: investigate Belgium's draws

[Karlis and Ntzoufras (2003)](https://doi.org/10.1111/1467-9884.00366) replace
independent score counts with a bivariate Poisson family and discuss diagonal inflation.
Their football example shows that even modest score dependence can affect fit and draw
prediction.

Disposition: **Experiment immediately after Dixon-Coles (E3)**.

This is especially relevant because the Belgium audit found weak draw recall and many
generic prediction vectors. Compare:

- independent Poisson;
- Dixon-Coles low-score correction;
- bivariate Poisson;
- diagonal-inflated bivariate Poisson.

The decision criterion is paired out-of-time log score/Brier/RPS plus draw calibration,
not “which model predicts the most draws.” A model can inflate draws and appear better
on recall while producing worse probabilities.

### 1.3 Dynamic latent strengths: replace repeated snapshots with state

[Koopman and Lit (2015)](https://doi.org/10.1111/rssa.12042) develop a dynamic
bivariate Poisson model whose intensity coefficients evolve stochastically over time.
The paper uses state-space and importance-sampling machinery and reports out-of-sample
Premier League results.

[Rue and Salvesen's dynamic Bayesian treatment](https://citeseerx.ist.psu.edu/document?doi=d260ef5bd7eedc2dd269453fab3507b5752536f1&repid=rep1&type=pdf)
likewise models time-varying offensive and defensive strength and retrospective changes.

Disposition: **P2/P3 experiment (E3)**.

Pitch Oracle does not need to reproduce the papers' inference machinery first. Begin
with an online Gaussian random-walk or score-driven update and compare it with periodic
Dixon-Coles refits. Promote only if the dynamic state improves early-season, manager-
change, promoted-team, and long-gap cohorts without becoming unstable.

### 1.4 Hierarchical Bayesian strength: useful, but watch overshrinkage

[Baio and Blangiardo (2010)](https://discovery.ucl.ac.uk/id/eprint/16040/) propose a
Bayesian hierarchical football model and explicitly address overshrinkage using a more
flexible mixture. The result is highly relevant to universal European leagues: partial
pooling can stabilize clubs with little top-flight history, but naïve shrinkage can erase
real differences.

Disposition: **Experiment for cold starts and uncertainty (E3)**.

Use cases:

- promoted/new clubs;
- early competition editions;
- league-specific home advantage drawn from a country-level distribution;
- attack/defense uncertainty propagated into match and season forecasts;
- transfer between related divisions without pretending they are equivalent.

Acceptance requires posterior predictive checks and out-of-time coverage. A credible
interval that is consistently too narrow is worse than an honest wide fallback.

### 1.5 Negative binomial, CMP, and Weibull counts: diagnose first

[Boshnakov, Kharrat, and McHale (2017)](https://doi.org/10.1016/j.ijforecast.2016.11.006)
use Weibull inter-arrival counts and a copula to permit richer marginal dispersion and
score dependence. The official Salford repository includes the
[accepted manuscript and abstract](https://salford-repository.worktribe.com/output/1395286/a-bivariate-weibull-count-model-for-forecasting-association-football-scores).
The open-source `goalmodel` implementation reviewed in the previous chapter also makes
negative-binomial and Conway-Maxwell-Poisson alternatives concrete.

Disposition: **Diagnostic-driven experiment (E2/E3)**.

Add a challenger only when residual reports show a failure it can express:

- negative binomial for overdispersed marginals;
- CMP for over- or underdispersion;
- hurdle/zero-inflated variants for unexplained zero excess;
- Weibull/copula when hazard/dispersion and dependence improvements justify operational
  complexity.

Do not reward in-sample AIC alone. The richer distribution must improve forecast scores
on held-out chronological folds and preserve fast, stable inference for all consumers.

### 1.6 Elo, Pi, and rating covariates: strong baselines, not magic probabilities

[Hvattum and Arntzen (2010)](https://doi.org/10.1016/j.ijforecast.2009.10.002) use
Elo ratings as covariates in ordered-logit match-result forecasts and compare several
benchmarks using statistical and economic measures.

[Constantinou and Fenton's Pi-rating paper](https://doi.org/10.1515/jqas-2012-0036)
updates home and away strength from score discrepancies and reported promising historical
performance. Results from one historical Premier League period do not establish a
current universal edge.

Disposition: **Adopt Elo; experiment with Pi/Glicko variants (E3)**.

Rating outputs must be calibrated before they are displayed as 1X2 probabilities. Use
them as:

- transparent baselines;
- opponent-strength adjustments;
- schedule difficulty;
- promoted-team priors;
- compact features for more flexible goals models.

### 1.7 Hybrid statistical/ML models: rank first, then add covariates

[Groll et al. (2019)](https://doi.org/10.1515/jqas-2018-0060) combine random forests
with Poisson-derived team ability parameters for international tournament forecasts.
The authors compare the hybrid with its building blocks. Later tournament work follows
similar combinations, but international tournaments differ materially from domestic
league forecasting.

Disposition: **Experiment, not priority (E2)**.

Pitch Oracle's version should predict home and away goal intensities/distributions,
not train an unconstrained 1X2 classifier that contradicts totals. Candidate covariates
must be point-in-time, and the rating-only and count-only models remain explicit
baselines. The current Belgium XGBoost/neural/LSTM artifacts do not gain legitimacy
merely because another paper used a random forest.

## 2. Player, shot, event, and tracking research

### 2.1 Shot-level expected goals

[Mead, O'Hare, and McMenemy (2023)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282295)
test richer xG feature sets and several algorithms across leagues. Their results reinforce
two useful points: shot context matters, and a more fashionable algorithm is not
automatically better—the paper reports strong logistic-regression performance in parts
of the study.

Disposition: **Capability-gated P2 experiment (E3)**.

Pitch Oracle requirements:

- shot location/angle/distance, body part, situation and other available pre-shot
  context;
- player/team features calculated as of the shot or trained strictly before the target
  fixture;
- league and provider calibration reports;
- explicit provider/model version on every xG artifact;
- no use of a post-shot feature such as outcome-adjacent information that would not be
  available at prediction time.

For pre-match forecasting, aggregate historical shot xG only up to the forecast cutoff.
Do not mix a provider xG definition and a locally trained xG definition in one unlabelled
trend line.

### 2.2 Player action values: xT and VAEP are analysis engines, not direct odds

[Decroos et al. (2019)](https://www.kdd.org/kdd2019/accepted-papers/view/actions-speak-louder-than-goals-valuing-player-actions-in-soccer)
introduce a common action language and the VAEP framework for valuing on-ball actions
through their effect on scoring/conceding probabilities. The associated
[`socceraction` repository](https://github.com/ML-KULeuven/socceraction) also implements
xT and VAEP variants.

Disposition: **Capability-gated product/research feature (E3)**.

Best product uses:

- team style and territorial threat profiles;
- player contribution/replacement priors for availability scenarios;
- “where threat came from” match review;
- recruitment/roster analysis only if the product scope expands.

Do not feed same-match action values into a pre-match forecast issued before the match.
Use only lagged, role/minute-adjusted player/team summaries. Do not label an aggregate
shot proxy as xT or VAEP.

### 2.3 Player-rating match forecasts

[Holmes and McHale (2024)](https://doi.org/10.1016/j.ijforecast.2023.03.002) forecast
match results using player abilities so team strength changes naturally with personnel.
[Arntzen and Hvattum (2021)](https://doi.org/10.1177/1471082X20929881) compare team
ratings and plus-minus-style player ratings in match outcome prediction.

Disposition: **P3 experiment once lineup/minute histories are reliable (E3)**.

Build a player-to-team prior, not a second app:

```text
historical player contribution
        × expected minutes
        × availability probability
        + replacement-level remainder
        → attack/defense lineup delta
        → existing score model
```

The lineup delta must shrink aggressively for low-minute players and unobserved leagues.
Pre-lineup forecasts integrate over lineup uncertainty; confirmed-lineup revisions are
new immutable forecast issues, never edits of the earlier forecast.

### 2.4 Tracking and pitch control

Pitch-control research estimates which player/team can reach and influence locations
on the pitch. Public educational code and small sample data make a prototype possible,
but broad synchronized tracking coverage is a commercial/data problem.

Disposition: **Research lab only (E2)**.

Potential UI:

- pitch-control surface at a selected frame;
- passing options gained/lost;
- defensive compactness and line height;
- off-ball value and space creation;
- tactical scenario examples.

These are genuinely new analyses, but they cannot be called universal until a provider
contract covers the target leagues. Core should provide the capability boundary now and
defer production promises.

## 3. Probabilistic forecast evaluation

### 3.1 Proper scores are non-negotiable

[Gneiting and Raftery (2007)](https://doi.org/10.1198/016214506000001437) explain
strictly proper scoring rules: they reward an honest predictive distribution rather
than a strategically distorted probability. Their
[author-hosted paper](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)
is a useful implementation reference.

Disposition: **Adopt (E4)**.

Pitch Oracle's mandatory scorecard:

- multiclass log loss/ignorance;
- multiclass Brier score;
- scoreline ignorance where a full grid exists;
- RPS as an ordinal football view;
- calibration/reliability by class and important cohort;
- sharpness/confidence distribution;
- market-baseline deltas when contemporaneous odds exist.

Accuracy, F1, draw recall, ROI, and calibration error may appear as secondary diagnostics.
None controls model promotion alone.

### 3.2 RPS is useful, but not the only truth

[Constantinou and Fenton (2012)](https://doi.org/10.1515/1559-0410.1418) argue that
home/draw/away are ordered and recommend Ranked Probability Score for football. A later
[case against making RPS the sole football evaluator](https://arxiv.org/abs/1908.08980)
disputes whether distance sensitivity is always desirable.

Disposition: **Adopt RPS in a metric panel (E3), reject RPS-only selection**.

This disagreement is productive. A forecast that moves probability from home to away
instead of home to draw may deserve a different ordinal penalty, while a user or market
decision can depend on all three quoted probabilities in other ways. Report RPS,
Brier, and log loss together and require improvements to be directionally consistent or
explicitly explained.

### 3.3 Calibration and sharpness need reproducible diagnostics

[Gneiting, Balabdaoui, and Raftery (2007)](https://doi.org/10.1111/j.1467-9868.2007.00587.x)
emphasize maximizing sharpness subject to calibration. More recently,
[Dimitriadis, Gneiting, and Jordan](https://arxiv.org/abs/2008.03033) propose CORP
reliability diagrams using isotonic regression to avoid arbitrary manual binning and
support score decomposition.

Disposition: **Adopt (E4)**.

Pitch Oracle should render:

- one-vs-rest CORP reliability curves for home/draw/away;
- uncertainty bands from resampling matchweeks/rounds rather than individual rows when
  temporal dependence is material;
- probability histograms to reveal a “calibrated” model that predicts only league
  priors;
- Brier decomposition or equivalent calibration/discrimination views;
- calibration by league, edition, forecast horizon, cold-start status, odds availability,
  and split/playoff phase.

ECE with ten fixed bins may remain a compact alert but cannot be the scientific evidence.

### 3.4 Calibration can matter more than accuracy for decisions

[Walsh and Joshi (2023)](https://arxiv.org/abs/2303.06021) compare accuracy-selected
and calibration-selected machine-learning models in an NBA betting experiment and
report better returns for calibration-selected models in their setting.

Disposition: **Supporting evidence only (E2)** because it is another sport and a
specific historical experiment. It strengthens, but does not prove, Pitch Oracle's
probability-first release gate.

## 4. Bookmaker odds as forecasts and benchmarks

### 4.1 The market is a serious baseline

[Štrumbelj and Robnik-Šikonja (2010)](https://doi.org/10.1016/j.ijforecast.2009.10.005)
evaluate odds from multiple online bookmakers across six European soccer leagues and
find differences by bookmaker and league, with forecasting effectiveness changing over
time.

Disposition: **Adopt a timestamped market baseline (E4)**.

Every odds-covered evaluation should include:

- each accepted bookmaker after de-vig;
- a documented consensus;
- best model versus consensus on identical fixtures;
- opener/current/closing snapshots kept separate;
- coverage and survivorship diagnostics;
- a no-market cohort so universal model quality remains visible.

The model is not “good” because it beats a class-prior baseline while losing materially
to available closing consensus.

### 4.2 Removing the overround is a modeling decision

[Štrumbelj (2014)](https://doi.org/10.1016/j.ijforecast.2014.02.008) compares methods
for converting odds to probability forecasts and reports that Shin-derived probabilities
outperformed basic normalization in the studied data, with differences varying by
market size and bookmaker.

Disposition: **Adopt multiple de-vig methods and backtest selection (E3)**.

Implement at least:

- multiplicative normalization;
- power transformation;
- Shin transformation;
- optional odds-ratio/additive methods if numerical constraints are explicit.

Never call raw inverse odds “probabilities” before removing the margin. Persist method,
overround, source, market timestamp, and convergence state.

### 4.3 Favorite-longshot bias is a cohort, not a free strategy

[Buhagiar, Cortis, and Newall (2018)](https://doi.org/10.1016/j.jbef.2018.01.010)
analyze more than 160,000 soccer odds across ten European leagues and report a
favorite-longshot pattern in their sample. Other studies find period/market variation.

Disposition: **Adopt the diagnostic cohort; reject a hard-coded betting rule (E3)**.

The Model Lab should plot calibration, price margin, return, and sample size by de-vigged
probability band. Any strategy based on this pattern must be selected before an untouched
test and must survive costs, source changes, and multiple-testing correction.

### 4.4 Market-informed and market-independent forecasts need separate identities

The market aggregates team news and information that may be difficult to source. A
market-aware residual model can be useful, but it answers a different question from an
independent football model.

Disposition: **Adopt two named tracks**:

- `independent`: no odds-derived feature in training or inference;
- `market_aware`: may use de-vigged probabilities, movement, or market-implied expected
  goals with explicit timestamps.

Never compare the market-aware model to the same market and call the overlap independent
predictive skill. Always retain the raw market baseline and an ablation without odds.

## 5. Betting evaluation and decision policy

### 5.1 ROI is not a forecast score

[Wunderlich and Memmert (2020)](https://doi.org/10.1016/j.ijforecast.2019.08.009)
show theoretically, through simulation, and with sports data that positive betting
returns can occur without superior forecast accuracy. Betting-return results are
sensitive to sample, bookmaker margin, price bands, and randomness.

Disposition: **Adopt as a governance rule (E4)**.

Pitch Oracle must never promote a forecast model solely because one strategy has
positive ROI. Economic evaluation comes after forecast evaluation and reports:

- bet count and unique fixtures;
- turnover, ROI, profit, maximum drawdown, and volatility;
- offered, executable, and closing prices with timestamps;
- commission, void/push/half-win settlement, and conservative slippage assumptions;
- flat stake and capped fractional policies;
- uncertainty intervals and longest losing run;
- results by league/season/market/price band;
- all thresholds tried, not only the winner.

### 5.2 Kelly assumes probabilities are right

[Kelly (1956)](https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf) connects
information and long-run capital growth under known probabilities. Sports models do not
know the true probability. Estimation error, calibration drift, simultaneous correlated
positions, limits, and finite user risk tolerance all violate idealized assumptions.

[Uhrín et al. (2021)](https://arxiv.org/abs/2107.08827) experimentally compare practical
sports-betting allocation variants and find value in risk-control modifications such as
adaptive fractional Kelly in their settings.

Disposition: **Education/research only, aggressively constrained (E3)**.

Default policy:

- zero stake if the edge confidence interval crosses zero;
- zero stake if price/forecast freshness fails;
- zero stake in an inadequately calibrated cohort;
- maximum quarter-Kelly after probability shrinkage toward the market;
- per-fixture, per-team, per-league, and total exposure caps;
- portfolio-aware handling of correlated outcomes;
- flat-stake results always reported beside Kelly results.

The UI must not imply guaranteed profit or personalize financial advice.

### 5.3 Asian handicap and quarter-line settlement must be exact

[Constantinou (2020)](https://arxiv.org/abs/2003.09384) studies Asian handicap market
efficiency with ratings and Bayesian networks across historical Premier League seasons.
Regardless of whether its inefficiency findings generalize, the market structure matters:
quarter lines split a stake over adjacent half/integer lines and can settle as full win,
half win, push, half loss, or full loss.

Disposition: **Adopt exact settlement and coherent grid pricing (E3)**.

Do not convert a 2.25 total into a made-up binary label. Store the line as an exact
decimal, split it mechanically, and calculate expected return from the joint score grid.

### 5.4 Data snooping is the central backtest risk

[White (2000)](https://doi.org/10.1111/1468-0262.00152) formalizes the danger of
reusing one history to select among many specifications and supplies a reality-check
framework against a benchmark. [Hansen's Superior Predictive Ability test
(2005)](https://doi.org/10.1198/073500105000000063) refines testing when many
alternatives are compared.

Disposition: **Adopt in the experiment platform (E4)**.

Pitch Oracle's practical controls:

1. preregister candidate families, primary metrics, cohorts, and thresholds in a
   versioned experiment spec;
2. fit/tune/calibrate only in nested historical windows;
3. keep one final forward season/time range untouched until the decision is frozen;
4. compare paired per-fixture loss differentials with block bootstrap intervals;
5. run a White-style/SPA family test when selecting the best of many candidates;
6. log failed and abandoned experiments;
7. require a shadow/live paper-trading period before any market-facing claim;
8. never keep scanning thresholds after seeing the held-out result.

## 6. What the papers do not establish

The research corpus does **not** establish that:

- Dixon-Coles always beats independent Poisson;
- bivariate Poisson always fixes draws;
- Bayesian models always calibrate better;
- player/event/tracking data improve pre-match forecasts enough to justify cost;
- a neural network is useful because it is more flexible;
- historical Premier League profit generalizes to Belgium, Turkey, the Netherlands,
  Scotland, another bookmaker, or current market microstructure;
- closing-line value proves eventual profit;
- fractional Kelly makes an incorrect edge safe;
- bookmaker consensus is the true probability;
- a selected open dataset has production licensing or target-league coverage.

These non-findings are part of the plan. They prevent the roadmap from presenting
research ideas as completed product claims.

## 7. Evidence-backed research initiatives

These 26 initiatives extend the 50-feature product catalog. They are experiments and
platform enablers, not 26 promises to deploy models.

| ID | Initiative | Evidence | Disposition | Promotion evidence |
|---|---|---:|---|---|
| R01 | Common `ScoreModel`/probability-grid API | E4 | Adopt P0/P1 | Parity and invariant tests for every candidate. |
| R02 | Count-distribution residual report | E4 | Adopt P1 | Per-league dispersion, zero, diagonal, and tail diagnostics. |
| R03 | Tuned recency/half-life | E4 | Adopt P1 | Nested rolling-fold gain over fixed/no decay. |
| R04 | Bivariate Poisson challenger | E3 | Experiment P1 | Paired proper-score and draw-calibration gain. |
| R05 | Diagonal-inflated/hurdle challenger | E2/E3 | Experiment P2 | Stable low-score improvement without global miscalibration. |
| R06 | NB/CMP dispersion challenger | E2/E3 | Experiment P2 | Residual need plus out-of-time score gain. |
| R07 | Weibull/copula challenger | E2/E3 | Defer P3 | Meaningful gain net of compute/maintenance. |
| R08 | Score-driven/state-space strengths | E3 | Experiment P2 | Early-season/change cohort and overall gain. |
| R09 | Hierarchical multi-league priors | E3 | Experiment P2 | Cold-start gain with calibrated intervals. |
| R10 | Elo/Pi/Glicko rating tournament | E3 | Adopt Elo; experiment others | Calibrated paired baseline comparison. |
| R11 | Rank-plus-covariate goals model | E2 | Experiment P2 | Beats rating and count components separately. |
| R12 | Player-strength lineup prior | E3 | Capability-gated P3 | Reliable point-in-time minutes plus ablation gain. |
| R13 | Independent vs market-aware tracks | E4 | Adopt P1/P3 | Separate registry IDs, features, and evaluation. |
| R14 | Multi-method de-vig engine | E3 | Adopt P2/P3 | Calibration by league/book/market and convergence tests. |
| R15 | Market-implied expected-goals comparator | E2/E3 | Experiment P3 | No leakage; residual value over market baseline. |
| R16 | Proper-score panel | E4 | Adopt P0 | Log/Brier/RPS/scoreline metrics from persisted OOF rows. |
| R17 | CORP reliability and sharpness view | E4 | Adopt P1 | Reproducible curves with uncertainty and cohort filters. |
| R18 | Paired block-bootstrap model deltas | E4 | Adopt P1 | Matchweek-aware confidence intervals. |
| R19 | White/SPA multi-candidate control | E4 | Adopt P2 | Family-level test recorded with experiment. |
| R20 | Immutable forward-test ledger | E4 | Adopt P0/P1 | Forecast fixed before kickoff; final set touched once. |
| R21 | Exact quarter-line settlement | E3 | Adopt P2/P3 | Exhaustive score/line golden tests. |
| R22 | Closing-price and CLV audit | E3 | Adopt P3 | Source/timestamp/coverage and de-vig method present. |
| R23 | Provider-neutral shot xG | E3 | Capability-gated P2 | Shot calibration and provider/version lineage. |
| R24 | xT/VAEP team/player snapshots | E3 | Capability-gated P3 | Stable canonical actions and lagged aggregation. |
| R25 | Tracking/pitch-control lab | E2 | Research only | Licensed representative coverage before production. |
| R26 | Uncertainty-aware capped portfolio simulator | E3 | Research only P3 | Zero-default gates, costs, correlation, and forward shadow test. |

## 8. Revised model promotion gate

A challenger may replace the champion only when all mandatory statements are true:

1. Inputs pass entity, observed-time, kickoff-time, edition, and leakage validation.
2. Fit and inference succeed on every required league or an explicit baseline fallback
   is exercised and reported.
3. The candidate improves the primary proper score in the frozen evaluation design.
4. Brier/RPS/log-loss changes are not materially contradictory without an approved
   explanation tied to product loss.
5. Calibration is not worse overall or for draws, cold starts, early season, and the
   target league.
6. Paired uncertainty excludes a practically harmful regression.
7. If many candidates were searched, family-level superiority survives the selected
   data-snooping test or the candidate remains a challenger pending forward evidence.
8. The candidate remains coherent across scoreline/1X2/totals/BTTS outputs.
9. Runtime, artifact size, failure rate, and reproducibility meet budgets.
10. No positive-ROI result is used as a substitute for the preceding gates.

Market-aware candidates add two more requirements:

11. They beat the corresponding de-vigged market baseline on the same rows.
12. Their ablation proves whether non-market information adds incremental value.

## 9. Belgium-specific research agenda

Belgium is a valuable proving ground because its current consumer exposes concrete
failure modes rather than an abstract benchmark:

1. Fix aliases, season/phase rules, and role-split history before running model research.
2. Freeze an immutable Belgium rolling-origin forecast ledger with current Poisson,
   class prior, and available market consensus.
3. Run R01–R04 first: common grid, residual report, tuned decay, bivariate challenger.
4. Report draw calibration and unique forecast-vector counts; do not optimize draw
   recall alone.
5. Test hierarchical priors on historical promoted/new teams, never only current clubs.
6. Verify Belgian regular-season and playoff/split phases as distinct evaluation cohorts.
7. Compare forecast performance before and after team-name resolution to quantify how
   much “model weakness” was actually entity failure.
8. Defer xG/player/action features until coverage and timestamps are audited; do not
   backfill current player states into old fixtures.
9. Treat odds as an optional Belgium capability. The independent model must run and
   remain evaluable without them.
10. Keep the final season segment untouched while the distribution tournament is tuned.

## 10. Research conclusion

The defensible way to make Pitch Oracle more interesting is not to add a “deep learning”
badge. It is to expose richer questions—score distributions, team-strength trajectories,
uncertainty, tactical/event value, lineup scenarios, season stakes, market disagreement—
while making every claim survive temporal, probabilistic, and operational scrutiny.

The literature supplies many credible candidates. It supplies no permission to skip
entity correctness, honest baselines, calibration, data-snooping control, or forward
validation. The next chapter turns these conclusions into copy-ready contracts and
experiment code.
