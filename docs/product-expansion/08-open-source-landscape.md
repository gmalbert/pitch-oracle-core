# Open-source football analytics landscape

This review asks a deliberately practical question: what should Pitch Oracle learn
from public football analytics repositories without turning core into a collection
of copied notebooks or fragile third-party wrappers?

The survey was performed on 2026-08-10. Repository descriptions, supported providers,
and maintenance status can change, so every dependency decision still requires a
fresh license, release, security, and compatibility check. Links below point to the
project repositories or their official documentation rather than secondary lists.

## Executive decisions

Adopt these patterns now:

1. Put every goals model behind one score-distribution interface and evaluate every
   candidate on identical rolling-origin rows.
2. Keep ingestion adapters, statistical forecasts, probability calibration, market
   conversion, and decision/staking policy as separate layers.
3. Make bookmaker consensus a formidable benchmark and optional input, never an
   undocumented ingredient in the independent forecast.
4. Introduce a provider-neutral event contract before adding shot xG, xT, VAEP, or
   player-value features.
5. Treat sample event/tracking repositories as adapter fixtures and research sandboxes,
   not as evidence of universal production coverage.
6. Benchmark selected calculations against mature open-source implementations, but
   retain Pitch Oracle's typed artifacts, point-in-time rules, and thin-consumer API.

Do not adopt these patterns:

- fixed market/model blend weights presented as universally optimal;
- random train/test splits over fixtures;
- a different forecasting pipeline for every market;
- automated bookmaker browser/account execution;
- scraping implementations without a source-policy, license, terms, and failure-mode
  review;
- event or tracking features silently replaced with made-up proxies;
- model promotion based on stars, benchmark screenshots, accuracy, or simulated ROI.

## Review rubric

Repositories were evaluated along seven dimensions:

| Dimension | Question |
|---|---|
| Forecast contract | Does the project emit a complete probability distribution or only a label/pick? |
| Time integrity | Can its examples be reproduced without future information? |
| Evaluation | Does it compare out of sample against credible simple and market baselines? |
| Universality | Can the idea work across domestic European competitions and league formats? |
| Data portability | Is provider-specific input converted to a stable internal representation? |
| Operational fit | Is the code packaged, tested, licensed, and plausibly maintainable? |
| Product fit | Does it help users understand a fixture, team, season, or forecast rather than merely add another algorithm? |

The result is an architectural recommendation, not a claim that a repository's
published model will reproduce its historical results on Belgium or another Pitch
Oracle league.

## 1. Penaltyblog: the closest modeling benchmark

[Penaltyblog](https://github.com/martineastwood/penaltyblog) is the most directly
comparable open-source project reviewed. Its public package spans Poisson,
Dixon-Coles, bivariate and other count models, hierarchical Bayesian variants,
multiple team-rating systems, implied-probability conversion, Asian handicap and
totals calculations, event-data processing, and forecast scoring. Its
[model documentation](https://penaltyblog.readthedocs.io/en/latest/models/overview.html)
also exposes a common API across several score models and supports time weighting.

What Pitch Oracle should borrow:

- a stable probability-grid result type shared by all goals models;
- a distribution tournament rather than a Poisson-versus-neural popularity contest;
- time weighting as a tunable, backtested model parameter;
- RPS, Brier, and ignorance/log scores in one evaluation report;
- derived-market calculations over the same grid, including quarter-line mechanics;
- several de-vig methods behind one typed interface;
- rating systems as both baselines and features.

What Pitch Oracle should not copy wholesale:

- Pitch Oracle's artifact and UI contracts serve a different product and remain the
  source of truth;
- provider access, package version support, compiled-extension deployment, and
  performance need a controlled spike before adding a runtime dependency;
- no package result substitutes for a league-by-league rolling-origin evaluation.

Recommended action: create a research-only parity suite for a fixed set of score
distributions and odds conversions. If `penaltyblog` passes compatibility and license
review, it may be used as an experiment backend or numerical oracle; production should
not become dependent on it merely to avoid implementing a small stable formula.

## 2. goalmodel: test the Poisson assumption, do not assume it

The R package [goalmodel](https://github.com/opisthokonta/goalmodel) implements
independent Poisson, negative-binomial, Conway-Maxwell-Poisson, Dixon-Coles, time
weights, scoreline/1X2/totals/BTTS predictions, expected-goal inference from market
probabilities, and multiple scoring rules.

Its most transferable lesson is diagnostic: football goals need not be equidispersed.
Negative binomial handles overdispersion; Conway-Maxwell-Poisson can handle under- or
overdispersion. That does not mean either should be deployed. It means each competition
edition should publish mean/variance, zero frequency, diagonal/draw residuals, and tail
residuals before selecting a count family.

Pitch Oracle action:

- add negative-binomial and CMP challengers only after a dispersion report says the
  Poisson restriction is materially wrong;
- compare these distributions on the exact same folds and probability grid;
- infer market expected goals only in the market-comparison track, not as a hidden
  feature in the no-odds champion;
- add fatigue/rest covariates through point-in-time snapshots, mirroring the project's
  useful previous-match-day helpers without copying role-split state.

## 3. socceraction: one event language, several value models

[socceraction](https://github.com/ML-KULeuven/socceraction) converts StatsBomb, Opta,
Wyscout, Stats Perform, and WhoScored event streams to SPADL/atomic-SPADL, then
implements Expected Threat (xT), VAEP, and Atomic-VAEP. This is strong support for a
provider-neutral event layer. It is not support for coupling the whole Pitch Oracle
product to SPADL or for advertising event features where no event provider exists.

The repository explicitly says it is primarily maintained for research
reproducibility rather than active feature development. That makes version pinning,
fixture snapshots, and local contract tests mandatory if it is used.

Pitch Oracle action:

```text
StatsBomb / Opta / Wyscout / other adapter
                    │
                    ▼
          CanonicalAction v1 contract
                    │
          ┌─────────┼──────────┐
          ▼         ▼          ▼
       shot xG      xT       VAEP-like value
          │         │          │
          └─────────┴──────────┘
                    ▼
       pre-kickoff team/player snapshots
```

The canonical action schema should preserve the raw provider event ID and payload
version so calculations can be reproduced. Event-derived features are capability
gated. A missing event feed means “not available,” never zero xT or zero VAEP.

## 4. soccer_xg: a reproducible xG research harness

[soccer_xg](https://github.com/ML-KULeuven/soccer_xg) is a Python package for training
and analyzing expected-goals models. It is valuable as a research reference because
it treats xG as a shot-level probability problem with data loading, feature extraction,
modeling, and analysis stages instead of treating “shots on target divided by a
constant” as xG.

Pitch Oracle action:

- reserve the `xg` field name for a calibrated shot-level probability sum from a
  documented model/provider;
- name current aggregate stand-ins `shot_quality_proxy` or similar;
- evaluate shot models by log loss/Brier and calibration, not shot classification
  accuracy;
- aggregate only shots observed before the forecast cutoff when constructing form;
- keep provider/model/version metadata alongside every match-level xG total.

## 5. StatsBomb open data and statsbombpy: a research corpus, not universal coverage

[statsbombpy](https://github.com/hudl/statsbombpy) reads StatsBomb's paid API and
selected open data. [StatsBomb Open Data](https://github.com/hudl/open-data) provides
competitions, matches, events, lineups, and selected 360 data under the associated
user agreement and attribution requirements.

This is ideal for:

- canonical event adapter development;
- xG, xT, and action-value reproducibility tests;
- UI prototypes for pass maps, shot maps, pressure, and freeze-frame context;
- golden files that prove a provider schema change is detected.

It is not a universal production feed. Open competition coverage is selected, and
the paid API is a commercial capability. Core therefore needs a `ProviderCapability`
contract and must never assume that because one open competition contains events,
every Belgian, Dutch, Turkish, Scottish, or future league fixture does.

## 6. soccerdata: adapters can replace some bespoke scripts

[soccerdata](https://github.com/probberechts/soccerdata) exposes readers for Club Elo,
ESPN, FBref, Football-Data.co.uk, Sofascore, SoFIFA, Understat, and WhoScored. The
project demonstrates the value of one reader API and local caching across heterogeneous
sources.

Pitch Oracle should run an adapter spike for the providers it already uses. The
decision is not “replace every fetch script.” It is:

1. compare historical coverage and field semantics for each target competition;
2. test rate limits, cache invalidation, schema drift, and failure behavior;
3. review source terms and the package license;
4. wrap any accepted reader behind Pitch Oracle's own normalized contract;
5. preserve raw snapshots and observed timestamps.

Even if no dependency is adopted, the project validates F47/F48's design: one provider
can fail without corrupting unrelated capabilities, and source caching should be
separate from normalized artifacts.

## 7. LaurieOnTracking and Metrica sample data: a future research lab

[LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
contains educational implementations for synchronized tracking/event data, player
velocities, formations, pitch control, expected possession value, and pass options.
[Metrica Sports sample data](https://github.com/metrica-sports/sample-data) supplies a
small set of anonymized synchronized matches in normalized coordinates.

Pitch Oracle action:

- use the sample matches as adapter fixtures and visual regression inputs;
- build a tracking-capability extension point, not a required core table;
- prototype pitch-control and off-ball-space views in a separate research profile;
- never fit a universal production model on a handful of sample matches;
- do not infer tracking-level concepts from box-score aggregates and label them as
  measured pitch control.

## 8. sports-betting: separate the forecaster from the bettor

[sports-betting](https://github.com/georgedouzas/sports-betting) organizes data
loaders, statistical/odds sources, estimator wrappers, bettor policies, backtests, and
a CLI as distinct concepts. That separation is exactly right for Pitch Oracle.

Pitch Oracle's equivalent should have five boundaries:

```text
provider observation → forecast → calibration → price comparison → decision policy
```

A model can improve forecast quality without generating an executable edge. A decision
policy can appear profitable because it searched many thresholds even when the model
has no durable advantage. Separate registries and ledgers make both statements
testable.

Do not adopt account-driving/browser automation. The reviewed repository itself warns
that such automation can conflict with bookmaker terms. Pitch Oracle should stop at
auditable education, research simulation, and user-exportable analysis unless a future
legal, responsible-gambling, platform, and security review explicitly expands scope.

## 9. football_predictions: market blending and a subtle UI correctness lesson

[football_predictions](https://github.com/DOsinga/football_predictions) combines an
Elo-derived expected-goals view with optional market information, tournament simulation,
and interactive controls. Two ideas are useful:

- a market-aware challenger can test whether independent football information adds
  anything beyond consensus;
- Python and browser/UI calculations need numerical parity tests.

One particularly important product rule follows from score grids: the most likely exact
score is not necessarily consistent with the most likely 1X2 class. Many draw scorelines
can each have high individual mass while the sum of all home-win scorelines is larger.
The Match Center must label these separately as “modal score” and “most likely outcome.”

Do not adopt a fixed 80/20 model/market mixture as a universal rule. Learn combination
weights only from past out-of-fold rows, cap them, and compare them with the unblended
market and no-market forecasts.

## 10. mplsoccer: richer visuals without provider lock-in

[mplsoccer](https://github.com/andrewRowlinson/mplsoccer) provides football-pitch
plotting and common visual primitives. It can accelerate static shot/pass maps and
share cards, but Pitch Oracle should first normalize coordinate systems and orientation.

Adoption rules:

- visualization receives canonical coordinates, never raw provider axes;
- chart subtitles disclose the event provider and coverage;
- empty data renders an availability state, not an empty pitch implying no actions;
- static exports must use the same filtered rows as the interactive view;
- visual code remains optional so score-only consumers do not import event packages.

## Cross-repository pattern matrix

| Pattern | Evidence | Pitch Oracle disposition |
|---|---|---|
| Common score-model API | Penaltyblog, goalmodel | Adopt now as `ScoreModel` + `ProbabilityGrid`. |
| Multiple count families | Penaltyblog, goalmodel | Experiment only after residual/dispersion diagnostics. |
| Recency weighting | Penaltyblog, goalmodel | Adopt and tune inside rolling folds. |
| Hierarchical Bayesian strengths | Penaltyblog | Experiment for promoted/cold-start teams. |
| Provider-neutral events | socceraction | Adopt the contract before provider features. |
| Reproducible shot xG | soccer_xg, StatsBomb | Capability-gated experiment. |
| xT/VAEP | socceraction | Team-style/player-impact extension, event data required. |
| Tracking/pitch control | LaurieOnTracking, Metrica | Research-only until licensed broad coverage exists. |
| Reader/cache abstraction | soccerdata | Adapter spike; wrap accepted readers. |
| Forecast/bettor separation | sports-betting | Adopt. No account automation. |
| Market/model blend | football_predictions | OOF challenger only; reject fixed universal weight. |
| Football-native graphics | mplsoccer | Optional UI dependency after coordinate normalization. |

## Build, borrow, or benchmark

| Capability | Build in core | Borrow dependency | Benchmark only |
|---|---:|---:|---:|
| Forecast/artifact contracts | Yes | No | No |
| Rolling-origin evaluator and ledger | Yes | No | Compare outputs |
| Basic Poisson/Dixon-Coles | Yes | Optional later | penaltyblog/goalmodel |
| Bivariate/CMP/Weibull families | Adapter | Candidate | penaltyblog/goalmodel |
| De-vig formulas | Yes | Optional numerical oracle | penaltyblog |
| Canonical event representation | Yes, minimal | Optional socceraction converter | SPADL fixtures |
| StatsBomb reader | Wrapper | statsbombpy | Open-data golden files |
| Multi-source score/stat readers | Wrapper | soccerdata after spike | Current custom fetchers |
| xT/VAEP | Adapter | socceraction when compatible | Published notebooks |
| Pitch plotting | Thin wrapper | mplsoccer | Static parity images |
| Betting account execution | No | No | No |

## Dependency intake checklist

No researched project enters production until this checklist is attached to a decision
record:

- exact commit/tag and release date;
- license and transitive-license review;
- supported Python/platform versions and wheel availability;
- maintainer activity and unresolved security advisories;
- deterministic golden-file comparison;
- runtime and artifact-size impact;
- source/provider terms and attribution requirements;
- cache and retry behavior;
- schema-drift test;
- removal/rollback plan;
- proof that the dependency does not leak odds or post-match data into the independent
  champion.

## Recommended spikes

### Spike OS-1 — score distribution parity

Fit independent Poisson and Dixon-Coles on a frozen Belgium fold using core and
`penaltyblog`. Compare expected goals, grid cells, tail mass, 1X2, totals, BTTS, and
quarter-line returns to `1e-8` where parameterization is equivalent. Differences must
be explained rather than rounded away.

### Spike OS-2 — dispersion tournament

On Belgium plus at least two structurally different leagues, compare Poisson,
Dixon-Coles, bivariate Poisson, negative binomial, and CMP. Do not include the current
season in hyperparameter selection. Report convergence failures and inference cost as
well as scores.

### Spike OS-3 — reader survivability

Compare core's ESPN, Club Elo, Understat, and MatchHistory paths with `soccerdata` for
90 days of scheduled runs. Measure fixture/team resolution, missingness, cache hits,
schema incidents, and recovery time. A shorter implementation is not a win if coverage
or auditability regresses.

### Spike OS-4 — canonical events

Convert one StatsBomb open match through `statsbombpy`/`socceraction` and through a
minimal native adapter. Persist both canonical results, validate coordinate direction,
event order, possession/team IDs, and shot outcomes, then choose the smallest durable
boundary.

### Spike OS-5 — tracking lab

Use the Metrica samples to render velocity, pitch-control, and pass-option outputs.
The deliverable is an optional contract and a data-requirements report, not a production
feature claim.

## Final open-source conclusion

Pitch Oracle is not missing one magic GitHub model. It is missing the machinery that
lets several sensible models compete honestly and lets richer data enter without
breaking every consumer. The researched repositories strongly support the planned
direction: normalize data once, produce one coherent probability distribution, score
it many ways out of time, separate forecast quality from betting policy, and add event
or tracking intelligence only behind explicit capabilities.
