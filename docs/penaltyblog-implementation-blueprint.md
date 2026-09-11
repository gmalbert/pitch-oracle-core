# penaltyblog Implementation Blueprint

This document turns the review into code-level suggestions. It is intentionally
written as implementation guidance rather than a patch: the next PR should
choose the smallest slice, add tests, and wire the resulting artifact into the
manifest only after the tests pass.

## 1. Put penaltyblog behind one score-model adapter

Pitch Oracle already has an internal `ScoreModel` protocol and a domain
`ProbabilityGrid` with explicit tail mass. `penaltyblog` should satisfy that
protocol through an adapter. The rest of the application should not need to
know whether a grid came from penaltyblog, an internal model, or a future
provider.

Suggested location: `pitch_oracle_core/models/penaltyblog_adapter.py`.

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from penaltyblog.models import DixonColesGoalModel

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid
from pitch_oracle_core.models.protocol import (
    FixtureFeatures,
    ForecastTrack,
    ModelSpec,
)


@dataclass
class PenaltyBlogDixonColes:
    """Pitch Oracle adapter; keep penaltyblog details at this boundary."""

    spec: ModelSpec
    model: DixonColesGoalModel | None = None

    def fit(self, matches: pd.DataFrame, *, cutoff_utc) -> "PenaltyBlogDixonColes":
        self.model = DixonColesGoalModel(
            goals_home=matches["goals_home"].to_numpy(copy=True),
            goals_away=matches["goals_away"].to_numpy(copy=True),
            teams_home=matches["team_home"].to_numpy(copy=True),
            teams_away=matches["team_away"].to_numpy(copy=True),
            weights=matches["recency_weight"].to_numpy(copy=True),
        )
        self.model.fit(use_gradient=True)
        return self

    def predict_grid(self, fixture: FixtureFeatures) -> ProbabilityGrid:
        if self.model is None:
            raise RuntimeError("model must be fitted before prediction")
        output = self.model.predict(
            fixture.home_team_id,
            fixture.away_team_id,
            max_goals=15,
            normalize=False,
        )
        mass = np.asarray(output.grid, dtype=float)
        represented = float(mass.sum())
        return ProbabilityGrid(
            mass=mass,
            tail_mass=max(0.0, 1.0 - represented),
            max_goals_home=mass.shape[0] - 1,
            max_goals_away=mass.shape[1] - 1,
        )


PRODUCTION_DC_SPEC = ModelSpec(
    model_id="pb-dixon-coles-v1",
    family="dixon_coles",
    track=ForecastTrack.INDEPENDENT,
    required_capabilities=frozenset({"historical_goals", "team_registry"}),
    hyperparameters={"xi": 0.0018, "max_goals": 15, "use_gradient": True},
)
```

Implementation notes:

- Pin and smoke-test the exact `penaltyblog` version because `output.grid`,
  normalization, and market helper semantics are part of the adapter contract.
- Keep `normalize=False` at the boundary so tail mass is visible to artifact
  validation. Normalize only when a downstream market explicitly allows it.
- Add a second adapter for Bayesian models rather than adding conditional
  branches to every consumer.
- Add `neutral_venue` to `FixtureFeatures` before using it in production. The
  upstream model supports it, and silently treating neutral venues as home
  venues is a systematic feature error.

## 2. Make walk-forward evaluation strict and schema-aware

The current helper should be hardened along these lines:

```python
def resolve_date_column(frame: pd.DataFrame, requested: str) -> str:
    for candidate in (requested, "datetime", "date", "kickoff_utc"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("matches require datetime, date, or kickoff_utc")


date_col = resolve_date_column(matches, date_col)
df = matches.assign(
    _evaluation_date=pd.to_datetime(matches[date_col], utc=True, errors="raise")
).sort_values("_evaluation_date", kind="stable").reset_index(drop=True)

for start in range(0, len(df) - window - horizon + 1, horizon):
    train = df.iloc[start : start + window]
    test = df.iloc[start + window : start + window + horizon]
    scored_rows = []
    errors = []
    for row_number, row in test.iterrows():
        try:
            scored_rows.append(score_one_fixture(model, row))
        except (KeyError, ValueError) as exc:
            errors.append({"row": row_number, "error": str(exc)})
    if not scored_rows:
        raise ValueError(f"fold {fold_idx} scored zero fixtures: {errors[:3]}")
    # Persist both n_test and n_scored, plus errors in a diagnostics artifact.
```

The production version should not silently drop a fixture. A recoverable source
problem can be recorded as an unavailable row, but a model or schema error
should fail the build. Fold output should include:

```text
fold_id, train_start, train_end, test_start, test_end,
n_train, n_test, n_scored, n_skipped, skip_reason_counts,
brier, log_loss, rps, tail_mass_mean, tail_mass_max
```

Use the existing `pitch_oracle_core.evaluation.rolling_origin` ledger for the
promotion artifact. The older helper is still useful for quick experiments,
but it should not be the only path that can declare a model better.

## 3. Replace the fake AIC report with two separate reports

The code currently calls the output an AIC leaderboard but records only model
parameter counts. Split it into:

### In-sample fit diagnostics

```python
def fit_diagnostics(model) -> dict[str, float | int | None]:
    log_likelihood = getattr(model, "loglikelihood", None)
    n_params = getattr(model, "n_params", None)
    aic = None
    if log_likelihood is not None and n_params is not None:
        aic = 2.0 * float(n_params) - 2.0 * float(log_likelihood)
    return {
        "n_params": int(n_params) if n_params is not None else None,
        "log_likelihood": float(log_likelihood) if log_likelihood is not None else None,
        "aic": aic,
    }
```

### Out-of-time promotion leaderboard

Use `score_panel()` or the canonical rolling-origin score functions and rank
models by the pre-declared primary score. Include paired bootstrap confidence
intervals, calibration error, tail mass, and cohort deltas. A model with a
better AIC but worse out-of-time log loss should remain a research challenger.

Rename `scripts/eval/aic_leaderboard.py` to
`scripts/eval/model_fit_diagnostics.py` if the report is intentionally limited
to in-sample fit.

## 4. Make Bayesian uncertainty real or explicitly experimental

The current `bayes_p_home_ci_width` is a heuristic. The production contract
should be one of these two choices:

### Preferred: posterior predictive draws

Create a version-specific adapter that consumes the model's posterior trace,
generates an outcome probability vector for each posterior draw, and returns
quantiles:

```python
def posterior_market_summary(draw_probabilities: np.ndarray) -> dict[str, float]:
    """Summarize posterior draws shaped (n_draws, 3)."""
    if draw_probabilities.ndim != 2 or draw_probabilities.shape[1] != 3:
        raise ValueError("posterior draws must have shape (n_draws, 3)")
    lower, median, upper = np.quantile(
        draw_probabilities, [0.05, 0.50, 0.95], axis=0
    )
    return {
        "p_home_mean": float(draw_probabilities[:, 0].mean()),
        "p_home_p05": float(lower[0]),
        "p_home_p50": float(median[0]),
        "p_home_p95": float(upper[0]),
        "p_home_ci_width": float(upper[0] - lower[0]),
        "p_draw_mean": float(draw_probabilities[:, 1].mean()),
        "p_away_mean": float(draw_probabilities[:, 2].mean()),
    }
```

The adapter must be tested against a pinned version and must persist MCMC
settings, convergence diagnostics, effective sample size, and the posterior
artifact hash. Do not use private internals without a version-specific contract
test.

### Safe fallback: research-only heuristic

If posterior predictive extraction is not ready, rename the field to
`bayes_p_home_heuristic_width`, put it under an experimental namespace, and
exclude it from production feature manifests and user-facing confidence copy.

## 5. Make training provenance complete

The model hash must cover every input that can change the fitted result. A
minimal canonical payload is:

```python
import hashlib
import json
import penaltyblog


def training_fingerprint(frame, *, hyperparameters, feature_policy_version):
    canonical = frame[
        ["date", "team_home", "team_away", "goals_home", "goals_away", "recency_weight"]
    ].copy()
    canonical["date"] = canonical["date"].astype("datetime64[ns]").astype(str)
    payload = {
        "rows": canonical.to_dict(orient="records"),
        "hyperparameters": hyperparameters,
        "feature_policy_version": feature_policy_version,
        "penaltyblog_version": penaltyblog.__version__,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
```

Persist alongside the model:

- `penaltyblog_version`
- Python version and platform
- optimizer and gradient settings
- time-decay base date and `xi`
- source artifact names and hashes
- entity-registry version
- feature-policy version
- training date range and row count
- model schema version
- model fingerprint

This belongs in the existing artifact manifest rather than only in a loose JSON
sidecar.

## 6. Choose one probability-grid contract

Recommended architecture:

```text
penaltyblog model.predict(...)
              |
              v
PenaltyBlogGridAdapter (normalize=False, preserve tail)
              |
              v
domain.ProbabilityGrid  ---> market helpers ---> artifacts/UI
```

Do not have UI code call `FootballProbabilityGrid` directly while other pages
call `goal_markets.py` or `domain.probability_grid.py`. The adapter should expose
the common operations Pitch Oracle actually promises:

- 1X2 with explicit tail handling
- exact-score probability
- BTTS bounds/value
- totals with under/push/over
- Asian handicap win/push/lose
- expected goals bounds
- expected points

Then delete or deprecate duplicate implementations only after golden fixtures
show identical results within a declared tolerance.

## 7. Use market-implied goals as a comparator, not as a hidden feature

The upstream package includes goal-expectancy inversion helpers. Add a research
artifact that records:

```text
fixture_id, issued_at, bookmaker_set, market,
devig_method, p_home, p_draw, p_away,
p_over_2_5, implied_home_xg, implied_away_xg,
solver_error, quote_coverage, quote_freshness
```

Use it for:

- model-vs-market residuals;
- detecting weak or stale quotes;
- prior-free market baselines;
- closing-line-value studies;
- a feature in the market-aware track only.

Never feed bookmaker-derived values into the independent forecast track. The
existing `ModelSpec` and `validate_fixture_track()` checks should remain the
enforcement point.

## 8. Make MatchFlow a real artifact pipeline

The current wrappers are a good start, but production ingestion needs:

1. A match metadata table that maps event-file IDs to competition, season,
   kickoff, teams, provider, and source revision.
2. Schema validation before feature extraction.
3. A source snapshot hash and retrieval timestamp.
4. A capability result when event data is absent or coverage is partial.
5. A manifest artifact for raw events, normalized events, and derived xT.

Suggested pipeline shape:

```python
events = (
    Flow.from_glob("data_files/statsbomb/open-data/data/events/*.json", optimize=True)
    .filter(where_equals("match_id", match_id))
)

normalized = normalize_statsbomb_events(events, entity_registry=registry)
shots = extract_shots(normalized)
passes = extract_passes(normalized)
xt = fit_or_load_xt(competition_id, season_id, passes, shots)

publish_artifact(
    name=f"xt_{competition_id}_{season_id}",
    dependencies=("normalized_events", "entity_registry"),
    producer_version=penaltyblog.__version__,
)
```

The actual source normalization functions are intentionally left to the
consumer because StatsBomb, Opta, and Wyscout have different event semantics.

## 9. Ratings integration rules

Use ratings in three distinct ways:

- **Pre-match feature:** only the latest rating whose `known_at` is earlier
  than the fixture issue time.
- **Descriptive dashboard:** full-season rankings are allowed, but clearly
  labeled as retrospective.
- **Model challenger:** compare rating-derived probabilities against the
  goal-model champion in the same rolling-origin folds.

For `build_combined_rankings()`, use the union of home and away teams:

```python
home = matches[home_col].astype(str)
away = matches[away_col].astype(str)
teams = sorted(set(home).union(away))
```

Persist both the snapshot table and update history. The feature join should be
an as-of join, never a season-level merge.

## 10. Testing and release gates

Add these tests before promoting more upstream features:

| Test | Purpose |
|---|---|
| `test_penaltyblog_version_contract.py` | Assert pinned version, public imports, grid fields, and fit/predict signatures |
| `test_penaltyblog_grid_adapter.py` | Preserve tail mass and common market values against golden fixtures |
| `test_walk_forward_contract.py` | Date fallback, final fold, scored-row counts, and fail-closed behavior |
| `test_training_fingerprint.py` | Changing dates, xi, optimizer, or package version changes the fingerprint |
| `test_bayesian_uncertainty_contract.py` | Posterior intervals are real, monotonic, finite, and diagnostic-gated |
| `test_rating_point_in_time.py` | No future results enter a pre-match rating feature |
| `test_matchflow_manifest.py` | Event source hashes, metadata joins, coverage, and dependencies are published |
| `test_market_track_isolation.py` | Independent models cannot receive odds-derived values |
| `test_model_promotion.py` | Promotion uses out-of-time scores and calibration, not AIC alone |

Release gate for any new penaltyblog-backed artifact:

```text
provider API contract passes
and source coverage is declared
and feature timestamps pass leakage audit
and rolling-origin score is no worse than champion beyond tolerance
and calibration/cohort gates pass
and artifact metadata is complete
and consumer smoke tests pass
```

