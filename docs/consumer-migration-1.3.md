# Consumer migration: Pitch Oracle Core 1.3

Version 1.3 changes the model and artifact contract. Do not reuse models,
`preprocessed_data.pkl`, `upcoming_predictions.csv`, or `cache_manifest.json`
created by an earlier release.

Consumers must run Python 3.12 or newer and identify their league with
`PITCH_ORACLE_LEAGUE` (or the reusable workflow's required `league_key` input).

## Why a clean rebuild is required

Earlier pipelines could train on post-match statistics, full-history aggregates,
and randomly selected future matches. They also reconstructed upcoming features
independently and silently padded or truncated them to fit a model. Version 1.3
uses a point-in-time feature policy, chronological validation, and an exact feature
contract shared by training and prediction generation.

## Upgrade sequence

1. Pin `pitch-oracle-core[consumer]` to `v1.3.0` in the consumer requirements and
   every workflow default.
2. Replace the consumer prediction-cache feature reconstruction with:

   ```python
   contract = FeatureContract.load("precomputed/preprocessed_data.pkl")
   features = build_upcoming_feature_matrix(historical, upcoming, contract)
   probabilities = model.predict_proba(features)
   predictions = build_prediction_frame(upcoming, probabilities)
   ```

3. Delete or overwrite generated model and cache artifacts through the normal CI
   pipeline. Do not hand-edit or pad feature matrices.
4. Run data preparation, model training, database precomputation, upcoming
   prediction generation, and manifest generation in one job with one core pin.
5. Run `python -m pytest -q` and `python scripts/verify_parity.py`.
6. Review chronological accuracy and multiclass log loss before promoting the new
   artifacts. Old random-split metrics are not comparable.

## Required prediction-cache columns

The shared cache builder emits the fixture fields plus:

- `HomeWin_Prob`, `Draw_Prob`, and `AwayWin_Prob`;
- `PredictedResult`;
- `Risk_Score`, `Confidence_Score`, and `Risk_Category`;
- `Recommendation`;
- `PredictionGeneratedAt`.

`build_cache_manifest` validates these fields and verifies that the ensemble model
width matches the feature contract.
