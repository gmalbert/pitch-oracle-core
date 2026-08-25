# penaltyblog Adoption — Comprehensive Priority List

This document synthesises the 18 penaltyblog roadmaps into a single,
ordered priority list. It is the execution view of
[`00-roadmap-index.md`](00-roadmap-index.md).

## How priorities are assigned

- **P0 — Correctness and trust.** Fixes Pitch Oracle defects visible
  today (entity coverage, time, season identity, missing calibration,
  hand-rolled math). Without P0 we ship richer wrong answers faster.
- **P1 — Product experience.** High-value user-facing improvements that
  depend on P0.
- **P2 — Modelling depth.** Things that make the existing forecasts
  better once the foundation is solid.
- **P3 — Optional intelligence.** Things that are great when their
  provider is available and gracefully absent otherwise.

Effort estimates are the same conventions used in roadmap 13
(1 engineer, full days, no reviewer wait time).

---

## P0 — Correctness and trust (do first)

| # | Item | Why it is P0 | Source | Days |
|---|------|--------------|--------|------|
| 0.1 | **`pitch_oracle_core/team_mappings.py` shared module** | Today `team_name_mapping.py` is incomplete for the Bundesliga 2 and Belgium; every scraper duplicates the reconciliation logic. The single shared dict unblocks P0.2–P0.4. | 01 §1.5 | 1 |
| 0.2 | **Replace `fetch_clubelo.py` with `pb.ClubElo`** | Normalised output, no extra reconciliation, fits the cache-first pattern. | 01 §1.1 | 0.5 |
| 0.3 | **Replace `fetch_understat_xg.py` with `pb.Understat`** | Closes 4 known bugs (headers, cookies, dropped forecast columns, no name normalisation). | 01 §1.2 | 0.5 |
| 0.4 | **Replace ad-hoc `football-data.co.uk` pull with `pb.FootballData`** | The current `combine_raw_data.py` has no rate limiting and inconsistent date typing. | 01 §1.3 | 0.5 |
| 0.5 | **Time-decay weights on every goal-model fit** | The biggest single modelling win: today's training loop treats a 2018 freak result the same as yesterday. All eight `pb.models` accept `weights=...`. | 03 §3.8 | 0.5 |
| 0.6 | **Swap `train_models.py` Poisson training for `pb.DixonColesGoalModel`** | Fixes the systematic under-pricing of 0-0, 1-0, 0-1, 1-1; ~50× faster. | 03 §3.1, §3.2 | 1.5 |
| 0.7 | **Replace bespoke implied-probability code with `pb.implied.calculate_implied`** | The current `track_predictions.py` silently uses multiplicative de-vig; the new code can pick the right method per market. | 06 §6.1 | 0.5 |
| 0.8 | **Metrics + calibration baseline** | We currently have no reliability diagram. `pb.metrics` (RPS, Brier, log-loss, calibration) gives us a starting point for governance (see [14-metrics-evaluation.md](14-metrics-evaluation.md)). | 14 §14.1 | 1.5 |
| 0.9 | **`team_name_mapping.py` deletion** | Once 0.1–0.4 land, the legacy module is dead code. Delete on the same PR that adds the new shared module. | 01 §1.7 | 0.25 |
| 0.10 | **Manifest v3 alignment** | The new scrapers and grid artifacts must register under the existing manifest so the cache-contract test passes. | 13 §13.2 (wave 7) | 1 |
| 0.11 | **Pinned dependency + CI wheel** | `penaltyblog` is a Cython build. Pin a wheel-published version; add to a CI matrix covering Linux, macOS, Windows. | 13 §13.5 (risk row 1) | 0.5 |

**P0 subtotal: 8.25 days.**

P0 exit gate: every forecast flowing through the pipeline is produced by
a `penaltyblog` goal model with time-decay weights, the implied
probabilities come from a documented `pb.implied` method, and the
`penaltyblog` test suite is green on the consumer CI.

---

## P1 — Product experience

| # | Item | Why it is P1 | Source | Days |
|---|------|--------------|--------|------|
| 1.1 | **`FootballProbabilityGrid` → `pitch_oracle_core/markets.py`** | Single object replaces every hand-rolled "compute the price of market X" snippet in `goal_markets.py`, `evaluate_poisson.py`, `predictions.py`. 30+ markets for free. | 11 §11.1, §11.7 | 2 |
| 1.2 | **Match Centre rewrite with `pb.viz.Pitch`** | Replaces ad-hoc Plotly snippets with a single, themed, provider-aware pitch. Affects 4 Streamlit pages. | 10 §10.1, §10.10 | 2 |
| 1.3 | **Multi-model AIC leaderboard in Model Lab** | Lets users see *why* Dixon-Coles is the production default. Surfaces the new NB/ZIP/BP/Weibull candidates without committing to them. | 03 §3.13 | 1.5 |
| 1.4 | **Walk-forward evaluation harness** | Replaces today's "train on everything, evaluate on everything". Required before any candidate can be promoted. | 03 §3.12 | 2 |
| 1.5 | **Elo-based form widget** | Replaces ad-hoc "last 5" with an Elo delta over 28/56 days. Drop-in into `analyze_team_form.py`. | 05 §5.1, §5.7 | 1 |
| 1.6 | **Massey/Colley/Pi ratings joined to feature store** | The "Power Rankings" page becomes a sortable leaderboard with four ratings side-by-side. | 05 §5.5, §5.6 | 2 |
| 1.7 | **Model-vs-Market leaderboard** | Per-fixture: model probability, de-vigged book probability, edge. Powers the new "Value Bets" page. | 06 §6.7 | 1.5 |
| 1.8 | **MatchFlow rewrite of `explore_statsbomb.py`** | Replaces eager `pd.read_json` over the full archive with a lazy stream that scales. | 02 §2.1 | 2 |
| 1.9 | **MatchFlow rewrite of `explore_wyscout.py`** | Same change for the Wyscout S3 bucket. | 02 §2.3, 12 §12.3 | 2 |
| 1.10 | **`xTModel` per (competition, season)** | Powers the "Expected Threat" tab and feeds per-action xT into the ML ensemble. | 09 §9.2, §9.3 | 3 |
| 1.11 | **Streamlit "Today's Bets" page** | First user-facing betting page. Surfaces arbitrage + value opportunities. | 07 §7.7, §7.8 | 1.5 |
| 1.12 | **Fixture detail page on the grid** | 30-line Streamlit page that lists every market for the selected fixture. The canonical match-detail view. | 11 §11.10 | 0.5 |
| 1.13 | **Bayesian tab in `model_optimization.py`** | First user-facing view of credible intervals. | 04 §4.1, §4.3 | 2 |
| 1.14 | **`scripts/events/matchflow_shots.py` + `matchflow_passes.py`** | Reusable per-event artifacts that the rest of the pipeline depends on. | 02 §2.4, 12 §12.6 | 1.5 |
| 1.15 | **Golden-master test fixtures** | First batch of golden-master JSON/parquet files for the new scrapers, models, and grids. | 18 §18.1 | 1.5 |

**P1 subtotal: 24 days.**

P1 exit gate: the Streamlit app exposes the new Match Centre, Expected
Threat, Today's Bets, and Model Lab tabs; the walk-forward harness is
the source of truth for promotion decisions; `penaltyblog` is the
default engine for any new artifact.

---

## P2 — Modelling depth

| # | Item | Why it is P2 | Source | Days |
|---|------|--------------|--------|------|
| 2.1 | **Negative Binomial / Bivariate Poisson / ZIP / Weibull Copula model variants** | Each is one line of fit; each can become a Model Lab challenger. | 03 §3.4–§3.7 | 1.5 |
| 2.2 | **Hierarchical Bayesian model for sparse leagues** | Best for Scotland, Belgium, and Eredivisie promoted sides. | 04 §4.4 | 2 |
| 2.3 | **Bayesian uncertainty features into the ML ensemble** | `bayes_p_home_ci_width` etc. let XGBoost learn when to shrink toward 50/30/20. | 04 §4.7 | 1.5 |
| 2.4 | **Per-bookmaker margin audit dashboard** | Quarterly report on which books are sharp. | 06 §6.6 | 1 |
| 2.5 | **Full Kelly / multi-Kelly / value-bet detection** | `pb.betting` wiring into the daily value scan. | 07 §7.1, §7.2, §7.3 | 2 |
| 2.6 | **Arbitrage detection** | Same scan, separate section. | 07 §7.4 | 1 |
| 2.7 | **In-play hedge helper** | `arbitrage_hedge` for existing positions. | 07 §7.5 | 1 |
| 2.8 | **Odds conversion pipeline** | American / fractional / decimal across the entire ingest. | 07 §7.6 | 0.5 |
| 2.9 | **StatsBomb open-data xT training pipeline** | Streamed, not eager. Combines [02] and [09]. | 12 §12.5 | 2 |
| 2.10 | **`pb.xt` per-action features in the ML ensemble** | `xt_added_per90` as a feature alongside the existing goal-model outputs. | 09 §9.6 | 1.5 |
| 2.11 | **Bayesian calibration harness** | Walk-forward Brier for the Bayesian posterior, compared to MLE. | 04 §4.6 | 2 |
| 2.12 | **Per-fixture summary artifact via MatchFlow** | A single parquet per match with event counts and totals; feeds the Match Centre header. | 12 §12.6 | 1.5 |
| 2.13 | **Drift monitor MVP** | PSI on a small set of features; alert if any window exceeds threshold. | 15 §15.1, §15.2 | 2 |
| 2.14 | **Cohort performance slices** | Promoted teams, derbies, short rest, early season. | 14 §14.7 | 2 |
| 2.15 | **xT comparison vs pretrained model** | Spearman ρ of player rankings. Sanity check the custom fit. | 09 §9.8 | 0.5 |
| 2.16 | **Custom theme for `pb.viz.Pitch`** | One Theme instance reused by every Streamlit page. | 10 §10.6 | 0.5 |

**P2 subtotal: 22 days.**

P2 exit gate: Model Lab shows champion + challengers; every promotion
decision has a paired chronological comparison; the ML ensemble has
both `pb.models` outputs and `pb.xt` features; Market Lab exists
behind a capability flag.

---

## P3 — Optional intelligence (capability-gated)

| # | Item | Why it is P3 | Source | Days |
|---|------|--------------|--------|------|
| 3.1 | **FPL workspace** | Only the EPL consumer can host it; useless elsewhere. | 08 (whole file) | 3 |
| 3.2 | **FPL mini-league dashboard** | 3 additional days on top of 3.1. | 08 §8.6, §8.9 | 3 |
| 3.3 | **Captain-pick advisor + price-change predictor** | FPL-only enhancements. | 08 §8.7, §8.8 | 1.5 |
| 3.4 | **Opta / paid StatsBomb ingest via MatchFlow** | Adapter for the API; only relevant to consumers with those contracts. | 12 §12.1, §12.2 | 2 |
| 3.5 | **Per-bookmaker margin audit (quarterly report)** | Only worthwhile once several books are stable. | 06 §6.6 | 1 |
| 3.6 | **Responsible staking UI** | Kelly fraction slider, education copy, drawdown cap. | 07 §7.8, §7.9 | 1 |
| 3.7 | **Hierarchical cross-league prior** | Shares information across leagues for promoted teams. | 04 §4.4 + 13 §13.9 | 3 |
| 3.8 | **In-play markets** | Combines live xG with the grid. | 13 §13.9 | 5 |
| 3.9 | **xT-based feature engineering (long tail)** | A bundle of new ensemble features (per-action xT, xT risk, xT by zone). | 09 §9.6, 13 §13.9 | 3 |
| 3.10 | **Cross-league comparison index** | Aggregates metrics across all consumer artifacts. | 17 §17.5 | 3 |
| 3.11 | **Additional ratings systems (Glicko-2, TrueSkill)** | Only worth doing if Elo/Massey/Colley/Pi are insufficient. | 05 (extension) | 2 |
| 3.12 | **Style fingerprint clustering** | The pretrained surface is good; per-league style labels are downstream. | (defer to product-expansion F20) | — |

**P3 subtotal: ~27.5 days** (3.12 deferred).

P3 exit gate: every P3 feature renders an honest "not available" state
when its provider is absent; no P3 feature blocks P0/P1/P2.

---

## Cross-cutting work (parallel to all priorities)

| # | Item | Why | Source | Days |
|---|------|-----|--------|------|
| X.1 | **Dependency split: `runtime` / `pipeline` / `diagnostics` / `experiments-neural`** | Today the consumer extra installs Torch + SHAP. The new scrapers/models don't need them. | product-expansion 06 §"Dependency changes" | 0.5 |
| X.2 | **Test surface expansion (21 files)** | The new features require new test files; mostly golden masters and contract tests. | 13 §13.4, 18 (whole file) | 3 |
| X.3 | **Documentation update** | README + product-expansion docs need to mention `penaltyblog` and the new artifacts. | 13 §13.7, §13.8 | 1 |
| X.4 | **GitHub Actions / consumer workflow update** | Schedule the new scrapers, the Bayesian trace, the xT model, the value scan. | 13 §13.2 (wave 7), 15 (whole file) | 1.5 |
| X.5 | **Per-consumer smoke test** | Belgium / Netherlands / Scotland / Turkey / EPL each get one end-to-end AppTest smoke test. | 18 §18.4 | 2 |
| X.6 | **Capability flags in manifest v3** | A consumer without an odds provider hides the Market Lab; a consumer without event data hides xT; etc. | 17 §17.3, product-expansion 06 | 1 |

**Cross-cutting subtotal: 9 days.**

---

## Roll-up

| Phase | Subtotal days |
|-------|---------------|
| P0 | 8.25 |
| P1 | 24.0 |
| P2 | 22.0 |
| P3 | 27.5 |
| Cross-cutting | 9.0 |
| **Total** | **~91 days** |

Roughly **18 engineer-weeks** end-to-end. Roadmap 13's 29.5-day
estimate covers only P0 + the easiest half of P1, which is the right
amount of work to do *before* the rest of the team can productively
contribute. The P2/P3 backlog is the multi-quarter roadmap.

---

## Sequencing rules

1. **P0 must complete before any P1 work merges.** The hand-rolled
   Poisson + ad-hoc scrapers make the P1 changes meaningless.
2. **P1.1 (markets) ships before P1.5/P1.6 (ratings) and P1.7
   (model-vs-market).** The grid is the dependency; the value-bet
   page can't be built without it.
3. **P1.4 (walk-forward harness) ships before P1.3 (AIC
   leaderboard).** Promotion criteria need a measurement instrument.
4. **P2.1 (new model variants) ships before P2.11 (Bayesian
   calibration).** You can't compare what you haven't run.
5. **P3 work is opt-in per consumer.** Belgium doesn't need an FPL
   page; the EPL consumer doesn't need a relegation-race simulator.

---

## Definition of done (P0 + P1)

- [ ] `team_name_mapping.py` is deleted; every consumer imports from
      `pitch_oracle_core/team_mappings.py`.
- [ ] `fetch_clubelo.py`, `fetch_understat_xg.py`, and
      `combine_raw_data.py` all route through `pb.scrapers`.
- [ ] `train_models.py` produces a `DixonColesGoalModel` pickle with
      time-decay weights and persists the model + a hash of the
      training set.
- [ ] `track_predictions.py` and the implied-probability code use
      `pb.implied.calculate_implied` with the LOGARITHMIC default.
- [ ] `walk_forward.py` and `aic_leaderboard.py` are part of the
      artifact pipeline and gate model promotion.
- [ ] Match Centre, Expected Threat, Today's Bets, Model Lab, and
      the Bayesian tab are live in at least one consumer (EPL).
- [ ] Every artifact file referenced by `manifest.json` exists and
      has the expected schema; `tests/test_artifacts.py` is green.
- [ ] CI runs `pip install penaltyblog` and the full test suite on
      Linux, macOS, and Windows.
- [ ] P0/P1 documentation update is merged in the README and the
      product-expansion docs.

---

## What is intentionally *not* in this list

- **A wholesale rewrite of the ML ensemble** (XGBoost / RF / GBT /
  LR). `penaltyblog` is a *parallel* set of models that the existing
  ensemble can ingest as features. The promotion gate (P1.4) is what
  protects the ensemble from regression.
- **A neural / LSTM extension.** `neural` is already an optional
  extra and the roadmaps don't depend on it.
- **A switch to `penaltyblog` for the *betting* math in
  `best_bets.py` / `risk.py` that already exists.** The P0/P1 plan
  brings the new math in *alongside* the old; P3 is the cutover
  point.
- **A new artifact format.** The new model outputs register under
  manifest v3, which is already the planned upgrade.
