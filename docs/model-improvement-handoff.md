# Pitch Oracle Model Improvement Handoff

## Superior-model review (2026-08-04)

The original diagnosis was directionally correct but not sufficient to ship a
trustworthy model. Two release-blocking issues were found in the implementation:

1. The old prematch policy admitted 196 numeric fields in the Netherlands
   dataset. About 151 are bookmaker or market-derived fields. Many closing-price
   columns use terse football-data codes such as `PSCH`, `AvgCA`, and `BFECAHH`,
   so excluding only names containing `Closing` did not remove them.
2. The Netherlands history contains completed fixtures dated after the audit
   date. Ambiguous day-first source dates were previously parsed as month-first;
   for example, 10 May can become 5 October. Any chronological split or rolling
   feature based on that file is invalid until the raw history is regenerated.

The immediate decision is therefore:

- Make the strict no-odds contract the production default.
- Block training when completed matches are dated in the future.
- Regenerate the Netherlands history from raw football-data files using the
  explicit source-date parser.
- Run the ablation only after that chronology gate passes.
- Calibrate on a dedicated middle period and evaluate once on a later untouched
  test period.

Implemented contracts:

- `pitch_oracle_core/features.py` owns market-field classification and the
  no-odds feature list (feature policy v3).
- `scripts/audit_model_features.py` writes `feature_inventory.csv` and
  `model_ablation.json` and refuses to report misleading metrics when chronology
  is invalid.
- `train_models.py` uses a chronological train/calibration/test partition,
  temperature scaling, and writes `model_metadata.json`.
- `precompute_database.py` uses the identical no-odds feature and imputation
  contract.

Run the gate and audit from the core repository:

```powershell
venv\Scripts\python.exe scripts\audit_model_features.py `
  C:\Users\gmalb\Downloads\netherlands-soccer\data_files\combined_historical_data_with_calculations_new.csv `
  --output-dir output\model-audit
```

Do not deploy a newly trained artifact unless all of these are true:

- Audit status is `complete`, with zero invalid dates and zero completed-future
  rows.
- The serialized metadata says `feature_set: no_odds` and feature policy v3.
- Test log loss and multiclass Brier score beat the class-frequency baseline.
- Calibration error is reported on the untouched test period and is no worse
  than the uncalibrated candidate.
- Upcoming feature imputation and duplicate-prediction rates are included in the
  release report; a repeated probability triplet must be traceable to genuinely
  identical no-odds feature rows.
- A betting recommendation is emitted only when matched, normalized market odds
  demonstrate a validated edge. Without odds, the output is a model lean, not a
  bet.

## Completed rebuild results (2026-08-04)

The Eredivisie history was regenerated from four available football-data season
files. The not-yet-published 2026–27 file returned HTTP 404 and was not treated
as data. The corrected dataset contains 1,224 completed fixtures from 5 August
2022 through 17 May 2026, with zero invalid dates, completed-future rows, or
duplicate fixtures.

Rolling-origin development results selected the 41-feature regularized no-odds
classifier:

| Candidate | Accuracy | Log loss | Brier | Calibration error |
|---|---:|---:|---:|---:|
| Class-prior baseline | 44.8% | 1.070 | 0.647 | 0.005 |
| Odds-heavy logistic | 46.2% | 1.363 | 0.699 | 0.152 |
| No-odds logistic | 51.4% | 1.031 | 0.614 | 0.068 |

On the final chronological holdout, the temperature-scaled production model
again cleared the release gate: 49.8% accuracy, 1.036 log loss, 0.623 Brier,
and 0.046 calibration error versus baseline log loss 1.082 and Brier 0.655.

The refreshed 63-fixture prediction cache has 63 unique probability triplets.
Provider aliases resolve every current club. Newly promoted ADO Den Haag still
uses prior/imputed model state until it accumulates Eredivisie history, while
its home-stadium coordinates are available for forecast weather.
The cache exposes 15 threshold-clearing model leans and 63 explicit `No bet`
decisions because no matched live odds are currently available.

## Objective

Improve the usefulness of betting recommendations. The current Eredivisie
Predictions page labels virtually every fixture `No Clear Edge`, making the
output less useful even though it still produces a highest-probability outcome.

The goal is not to manufacture more bets. The goal is to produce calibrated,
out-of-sample probabilities and identify genuine betting value when market
odds are available.

## Current diagnosis

The current model has a train/serve mismatch:

- Historical training data contains extensive bookmaker-odds features.
- Upcoming fixtures do not contain bookmaker odds.
- Missing upcoming features are filled with training-period imputation values.
- This compresses predictions toward a near-uniform 1X2 distribution.

Observed in the current Netherlands cache:

- 63 upcoming fixtures.
- Highest predicted win probability: approximately 53.2%.
- The guidance engine requires at least 55% for `Consider` and 65% for
  `Strong` recommendations.
- Only 5 fixtures have risk scores at or below 80.
- No fixture satisfies both the probability and risk thresholds.
- 34 fixtures have duplicate probability triplets.

Therefore, the guidance logic is conservative but behaving as designed. The
larger problem is that the model is receiving many imputed odds-related inputs
at inference time.

## Existing relevant code

- `train_models.py` — trains the XGBoost and ensemble models.
- `pitch_oracle_core/features.py` — defines the prematch feature policy.
- `pitch_oracle_core/predictions.py` — builds prediction artifacts.
- `pitch_oracle_core/risk.py` — calculates risk and betting guidance.
- `pitch_oracle_core/ui_pages.py` — renders the Predictions page.
- `templates/consumer/scripts/precompute_predictions.py` — consumer artifact
  generation.
- `docs/06-odds-coverage-plan.md` — existing live-odds integration plan.

## Recommended architecture

### 1. Build a true no-odds model

Train a separate model using only features that are available before kickoff
without bookmaker data:

- Rolling xG and goal difference.
- Recent form and points.
- Home/away performance splits.
- Team strength ratings such as ClubElo.
- Rest days and fixture congestion.
- Historical H2H features, if point-in-time safe.
- Injuries, suspensions, and expected lineups.
- Weather and venue conditions.

Exclude bookmaker odds, implied probabilities, bookmaker margins, value
features, and closing-price features from this model.

Use this model whenever upcoming odds are unavailable. This eliminates the
current imputation-driven collapse toward 33/33/33.

### 2. Add live odds as a separate input/source

When odds are available:

- Keep the football model probability independent from the market.
- Convert odds into normalized implied probabilities.
- Compare model probability against market probability after accounting for
  bookmaker margin.
- Recommend a bet only when the model has a sufficient edge and acceptable
  uncertainty.

The existing `docs/06-odds-coverage-plan.md` recommends Odds-API.io for the
new league consumers. The implementation should preserve graceful fallback to
the no-odds model if the API is unavailable.

### 3. Calibrate probabilities

Raw classifier probabilities should be calibrated using strictly out-of-sample
predictions. Evaluate:

- Multiclass log loss.
- Multiclass Brier score.
- Reliability/calibration curves.
- Expected calibration error.

Consider temperature scaling or isotonic calibration, fitted only on a
validation period that is later than the training period.

### 4. Separate model lean from betting recommendation

The UI should distinguish these concepts:

- **Model lean:** the highest-probability 1X2 outcome.
- **Bet recommendation:** a wager supported by both model confidence and
  market value.

Example:

```text
Model lean: Go Ahead Eagles
Bet recommendation: No bet — model edge versus market is insufficient
```

This lets the app communicate a useful forecast without presenting every lean
as a profitable wager.

### 5. Blend complementary models

Evaluate a blend of:

- No-odds ML model.
- Poisson expected-goals model.
- Market-implied probabilities, when available.

Do not assume equal weights. Select weights using chronological validation and
freeze them for deployment until the next scheduled retraining cycle.

## Guidance threshold recommendations

Do not simply lower the 55% threshold to create more recommendations. That
would increase the number of displayed picks without demonstrating more value.

Instead:

- Keep `Model lean` available at any confidence level.
- Define `Consider` and `Strong` thresholds from historical calibration and
  betting backtests.
- Require a market edge for an actual bet recommendation when odds exist.
- Report `No bet` when the forecast is plausible but lacks demonstrated value.

## Validation plan

Run chronological, rolling-origin backtests. Do not rely on random splits.

Compare these candidates:

1. Current odds-heavy model with imputed upcoming odds.
2. No-odds model.
3. Poisson/xG model.
4. No-odds plus Poisson blend.
5. Blend including market probabilities when available.

For each candidate, report:

- Accuracy.
- Log loss.
- Brier score.
- Calibration error.
- Draw recall and draw calibration.
- Percentage of fixtures with an actionable lean.
- Percentage of fixtures with a bet recommendation.
- ROI after bookmaker margin, if odds and historical closing prices are
  available.
- Maximum drawdown and number of bets.

The primary selection criteria should be out-of-sample log loss, calibration,
and risk-adjusted betting value—not the raw count of recommendations.

## Suggested implementation phases

### Phase 1 — Audit and ablation

- Inventory every training feature and classify it as odds, team form, xG,
  schedule, injury, weather, or other.
- Quantify missingness and imputation rates for upcoming fixtures.
- Detect duplicate feature rows and duplicate probability rows.
- Produce chronological out-of-sample metrics for odds versus no-odds feature
  sets.

### Phase 2 — No-odds production model

- Define and persist a no-odds feature contract.
- Train and serialize a no-odds model.
- Add explicit model metadata to the prediction artifact.
- Use the no-odds model whenever odds coverage is incomplete.

### Phase 3 — Probability calibration

- Generate rolling validation probabilities.
- Fit and persist a calibrator.
- Add calibration metrics to Model Lab.

### Phase 4 — Live market comparison

- Implement the selected odds adapter from the existing odds coverage plan.
- Normalize odds and remove overround.
- Match odds to fixtures using normalized team names and kickoff time.
- Calculate model-versus-market edge.

### Phase 5 — Recommendation redesign

- Show model lean separately from bet recommendation.
- Add explicit `No bet` explanations.
- Use thresholds selected from historical validation rather than arbitrary
  percentages.

### Phase 6 — Blend and monitor

- Compare the no-odds, Poisson, and market-aware models.
- Select blend weights chronologically.
- Track calibration, drift, duplicate predictions, missingness, and realized
  performance after each refresh.

## Important implementation cautions

- Do not use current/final league tables or full-season aggregates for past
  historical rows; this creates leakage.
- Do not train with closing odds if forecasts are generated before closing.
- Do not use live odds as training features unless the same timing and feature
  definition are available at inference.
- Do not lower thresholds solely to make the UI look more actionable.
- Preserve a graceful no-odds fallback when the live odds provider fails.

## Next release step

Publish the completed no-odds model contract as core v1.3.16, regenerate the
consumer prediction cache with forecast weather, and validate the refreshed
consumer application before enabling any live-odds integration.
