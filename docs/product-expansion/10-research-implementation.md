# Evidence-backed research implementation

This chapter is copy-ready scaffolding for initiatives R01–R26. It extends the code in
`04-modeling-implementation.md` and `07-context-and-market-implementation.md`; it does
not replace their domain, artifact, or UI contracts.

The most important implementation rule is separation:

```text
historical observations
        │
        ▼
independent forecast ───────┐
                            ├── proper-score evaluation
market observation ─────────┤
        │                   ├── calibrated comparison
        ▼                   └── decision-policy simulation
de-vigged market forecast
```

Odds never silently enter the independent track. A profitable decision-policy backtest
never silently promotes the forecast that fed it.

## Proposed code map

```text
pitch_oracle_core/
  domain/
    probability_grid.py
    research.py
  models/
    protocol.py
    independent_poisson.py
    distribution_registry.py
    hierarchical_prior.py
  evaluation/
    scores.py
    distribution_diagnostics.py
    calibration.py
    rolling_origin.py
    paired_tests.py
    experiment_registry.py
  markets/
    devig.py
    settlement.py
    benchmark.py
  events/
    schema.py
    capabilities.py
  players/
    lineup_strength.py
  ui/pages/
    research_lab.py
```

These modules belong in core. League consumers provide configuration and artifacts,
not alternate implementations.

## 1. A rigorous probability-grid contract (R01, R21)

The current `MatchForecast` example normalizes a finite matrix to one. Research models
need to expose truncation rather than hide it. Use a grid whose represented cells plus
tail equal one, then require a tiny tail before emitting point-valued markets.

```python
# pitch_oracle_core/domain/probability_grid.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ProbabilityGrid:
    """Joint home/away score mass with explicit unrepresented tail."""

    mass: np.ndarray
    tail_mass: float
    max_goals_home: int
    max_goals_away: int

    def __post_init__(self) -> None:
        values = np.asarray(self.mass, dtype=float)
        if values.ndim != 2 or values.size == 0:
            raise ValueError("mass must be a non-empty 2D array")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("mass must be finite and non-negative")
        if not np.isfinite(self.tail_mass) or not 0 <= self.tail_mass < 1:
            raise ValueError("tail_mass must be in [0, 1)")
        if values.shape != (self.max_goals_home + 1, self.max_goals_away + 1):
            raise ValueError("declared maxima do not match mass shape")
        if not np.isclose(values.sum() + self.tail_mass, 1.0, atol=1e-10):
            raise ValueError("represented mass plus tail must equal one")
        object.__setattr__(self, "mass", values)

    @property
    def represented_mass(self) -> float:
        return float(self.mass.sum())

    def outcome_lower_bounds(self) -> np.ndarray:
        # Rows are home goals; columns are away goals.
        return np.array(
            [
                np.tril(self.mass, k=-1).sum(),
                np.trace(self.mass),
                np.triu(self.mass, k=1).sum(),
            ],
            dtype=float,
        )

    def outcome_intervals(self) -> tuple[np.ndarray, np.ndarray]:
        lower = self.outcome_lower_bounds()
        # The tail's score relationship is unknown. Each marginal upper bound may
        # include all tail mass; these intervals are not meant to sum to one.
        upper = np.minimum(1.0, lower + self.tail_mass)
        return lower, upper

    def normalized_one_x_two(self, max_tail: float = 1e-8) -> np.ndarray:
        if self.tail_mass > max_tail:
            raise ValueError(
                f"tail {self.tail_mass:.3g} exceeds point-market limit {max_tail:.3g}"
            )
        values = self.outcome_lower_bounds()
        return values / values.sum()

    def expected_goals_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        home = np.arange(self.mass.shape[0], dtype=float)[:, None]
        away = np.arange(self.mass.shape[1], dtype=float)[None, :]
        lower = (
            float((self.mass * home).sum()),
            float((self.mass * away).sum()),
        )
        # Conservative finite upper bounds are impossible without a declared tail
        # support. Artifacts therefore publish lower estimates plus tail separately.
        return lower, (float("inf"), float("inf"))

    def exact_score_probability(self, home_goals: int, away_goals: int) -> float:
        if home_goals < 0 or away_goals < 0:
            return 0.0
        if home_goals >= self.mass.shape[0] or away_goals >= self.mass.shape[1]:
            return 0.0
        return float(self.mass[home_goals, away_goals])

    def btts_lower_bound(self) -> float:
        return float(self.mass[1:, 1:].sum())
```

Artifacts store `mass` as float32 only after calculations and validation. Evaluation
uses float64. A UI may render 0–6 while the model retains 0–12; visual truncation and
model truncation are different concepts.

### Independent Poisson adapter

```python
# pitch_oracle_core/models/independent_poisson.py
from math import exp, lgamma, log
import numpy as np

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid


def poisson_pmf(goals: int, rate: float) -> float:
    if goals < 0 or rate <= 0 or not np.isfinite(rate):
        raise ValueError("goals must be non-negative and rate finite/positive")
    return exp(goals * log(rate) - rate - lgamma(goals + 1))


def independent_poisson_grid(
    home_rate: float,
    away_rate: float,
    *,
    tail_tolerance: float = 1e-10,
    initial_max_goals: int = 8,
    hard_max_goals: int = 30,
) -> ProbabilityGrid:
    for maximum in range(initial_max_goals, hard_max_goals + 1):
        home = np.array([poisson_pmf(g, home_rate) for g in range(maximum + 1)])
        away = np.array([poisson_pmf(g, away_rate) for g in range(maximum + 1)])
        mass = np.outer(home, away)
        represented = float(mass.sum())
        tail = max(0.0, 1.0 - represented)
        if tail <= tail_tolerance:
            # Repair only machine-scale overshoot, not meaningful tail mass.
            if represented > 1.0:
                mass = mass / represented
                tail = 0.0
            else:
                tail = 1.0 - represented
            return ProbabilityGrid(mass, tail, maximum, maximum)
    raise RuntimeError("score grid did not meet tail tolerance before hard maximum")
```

## 2. One protocol for every candidate (R01, R13)

Candidates declare their track and required capabilities. The registry rejects a model
that claims to be independent while requesting odds features.

```python
# pitch_oracle_core/models/protocol.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Mapping, Protocol, Sequence
import pandas as pd

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid


class ForecastTrack(StrEnum):
    INDEPENDENT = "independent"
    MARKET_AWARE = "market_aware"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    track: ForecastTrack
    required_capabilities: frozenset[str]
    hyperparameters: Mapping[str, object]

    def validate(self) -> None:
        market_inputs = {"odds_1x2", "odds_totals", "odds_handicap", "market_movement"}
        if self.track == ForecastTrack.INDEPENDENT:
            forbidden = self.required_capabilities.intersection(market_inputs)
            if forbidden:
                raise ValueError(f"independent model requests market inputs: {forbidden}")


@dataclass(frozen=True)
class FixtureFeatures:
    fixture_id: str
    kickoff_utc: datetime
    home_team_id: str
    away_team_id: str
    values: Mapping[str, float | int | str | None]


class ScoreModel(Protocol):
    spec: ModelSpec

    def fit(self, matches: pd.DataFrame, *, cutoff_utc: datetime) -> "ScoreModel": ...

    def predict_grid(self, fixture: FixtureFeatures) -> ProbabilityGrid: ...


def validate_candidate_specs(specs: Sequence[ModelSpec]) -> None:
    seen: set[str] = set()
    for spec in specs:
        spec.validate()
        if spec.model_id in seen:
            raise ValueError(f"duplicate model_id: {spec.model_id}")
        seen.add(spec.model_id)
```

Do not create a `BelgiumDixonColesModel`. League identity, edition, home-advantage
prior, and capabilities are data/configuration passed to the same model.

## 3. Distribution diagnostics before distribution selection (R02, R04–R07)

The report below is intentionally simple and auditable. Add randomized quantile
residuals and posterior predictive checks for advanced models later.

```python
# pitch_oracle_core/evaluation/distribution_diagnostics.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GoalDiagnostics:
    fixtures: int
    home_mean: float
    home_variance: float
    away_mean: float
    away_variance: float
    home_dispersion_ratio: float
    away_dispersion_ratio: float
    zero_zero_rate: float
    draw_rate: float
    low_score_rate: float
    four_plus_goal_rate: float


def describe_goal_counts(matches: pd.DataFrame) -> GoalDiagnostics:
    required = {"home_goals", "away_goals"}
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    frame = matches.dropna(subset=list(required)).copy()
    if len(frame) < 2:
        raise ValueError("at least two completed fixtures are required")
    home = frame.home_goals.astype(int).to_numpy()
    away = frame.away_goals.astype(int).to_numpy()
    if (home < 0).any() or (away < 0).any():
        raise ValueError("goals cannot be negative")
    home_mean = float(home.mean())
    away_mean = float(away.mean())
    return GoalDiagnostics(
        fixtures=len(frame),
        home_mean=home_mean,
        home_variance=float(home.var(ddof=1)),
        away_mean=away_mean,
        away_variance=float(away.var(ddof=1)),
        home_dispersion_ratio=float(home.var(ddof=1) / max(home_mean, 1e-12)),
        away_dispersion_ratio=float(away.var(ddof=1) / max(away_mean, 1e-12)),
        zero_zero_rate=float(((home == 0) & (away == 0)).mean()),
        draw_rate=float((home == away).mean()),
        low_score_rate=float(((home <= 1) & (away <= 1)).mean()),
        four_plus_goal_rate=float(((home >= 4) | (away >= 4)).mean()),
    )


def diagnostics_by_edition(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (competition_id, edition_id), group in matches.groupby(
        ["competition_id", "edition_id"], sort=True
    ):
        report = asdict(describe_goal_counts(group))
        rows.append(
            {"competition_id": competition_id, "edition_id": edition_id, **report}
        )
    return pd.DataFrame(rows)
```

Interpretation is not a hard threshold such as “ratio 1.1 means negative binomial.”
Use the report to justify a frozen candidate set; use rolling-origin proper scores to
choose among it.

## 4. Proper-score panel (R16)

Use one canonical outcome order everywhere. This example uses home, draw, away.

```python
# pitch_oracle_core/evaluation/scores.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


HOME, DRAW, AWAY = 0, 1, 2


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("probabilities must have shape (n, 3)")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("each probability row must sum to one")
    return values


def outcome_index(home_goals: np.ndarray, away_goals: np.ndarray) -> np.ndarray:
    home = np.asarray(home_goals)
    away = np.asarray(away_goals)
    return np.where(home > away, HOME, np.where(home == away, DRAW, AWAY)).astype(int)


def multiclass_log_loss(y: np.ndarray, probabilities: np.ndarray) -> float:
    p = validate_probabilities(probabilities)
    labels = np.asarray(y, dtype=int)
    if labels.shape != (len(p),) or not np.isin(labels, [HOME, DRAW, AWAY]).all():
        raise ValueError("invalid outcome labels")
    selected = np.clip(p[np.arange(len(p)), labels], 1e-15, 1.0)
    return float(-np.log(selected).mean())


def multiclass_brier(y: np.ndarray, probabilities: np.ndarray) -> float:
    p = validate_probabilities(probabilities)
    observed = np.eye(3, dtype=float)[np.asarray(y, dtype=int)]
    return float(np.square(p - observed).sum(axis=1).mean())


def ranked_probability_score(y: np.ndarray, probabilities: np.ndarray) -> float:
    p = validate_probabilities(probabilities)
    observed = np.eye(3, dtype=float)[np.asarray(y, dtype=int)]
    # Only the first K-1 cumulative terms are required.
    forecast_cdf = np.cumsum(p, axis=1)[:, :-1]
    observed_cdf = np.cumsum(observed, axis=1)[:, :-1]
    return float(np.square(forecast_cdf - observed_cdf).sum(axis=1).mean() / 2.0)


@dataclass(frozen=True)
class ScorePanel:
    fixtures: int
    log_loss: float
    brier: float
    rps: float
    accuracy: float
    draw_recall: float


def score_panel(y: np.ndarray, probabilities: np.ndarray) -> ScorePanel:
    p = validate_probabilities(probabilities)
    labels = np.asarray(y, dtype=int)
    prediction = p.argmax(axis=1)
    draw = labels == DRAW
    return ScorePanel(
        fixtures=len(labels),
        log_loss=multiclass_log_loss(labels, p),
        brier=multiclass_brier(labels, p),
        rps=ranked_probability_score(labels, p),
        accuracy=float((prediction == labels).mean()),
        draw_recall=float((prediction[draw] == DRAW).mean()) if draw.any() else float("nan"),
    )
```

Lower is better for log loss, Brier, and RPS. Accuracy/draw recall are diagnostics and
may not be used to repair a model by distorting probabilities.

### Scoreline ignorance with explicit tail

```python
from math import log

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid


def scoreline_ignorance(
    grid: ProbabilityGrid,
    observed_home: int,
    observed_away: int,
    *,
    floor: float = 1e-15,
) -> float:
    if observed_home <= grid.max_goals_home and observed_away <= grid.max_goals_away:
        probability = grid.exact_score_probability(observed_home, observed_away)
    else:
        # The tail is one aggregate event. Models with repeated tail observations
        # should increase support rather than game this category.
        probability = grid.tail_mass
    return -log(max(probability, floor))
```

## 5. Reproducible calibration curves (R17)

The following CORP-style curve uses isotonic regression and groups equal fitted levels.
Bootstrap the complete calculation by chronological block for uncertainty bands.

```python
# pitch_oracle_core/evaluation/calibration.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class ReliabilityPoint:
    forecast_mean: float
    observed_rate: float
    fitted_rate: float
    count: int


def corp_reliability_curve(
    forecast_probability: np.ndarray,
    observed: np.ndarray,
) -> list[ReliabilityPoint]:
    probability = np.asarray(forecast_probability, dtype=float)
    truth = np.asarray(observed, dtype=float)
    if probability.shape != truth.shape or probability.ndim != 1:
        raise ValueError("forecast and observed must be equal-length vectors")
    if len(probability) < 2 or (probability < 0).any() or (probability > 1).any():
        raise ValueError("invalid probability vector")
    if not np.isin(truth, [0.0, 1.0]).all():
        raise ValueError("observed values must be binary")

    order = np.argsort(probability, kind="mergesort")
    p_sorted = probability[order]
    y_sorted = truth[order]
    fitted = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    ).fit_transform(p_sorted, y_sorted)

    # Consecutive equal PAV levels form reproducible blocks.
    boundaries = np.r_[0, np.flatnonzero(np.diff(fitted) != 0) + 1, len(fitted)]
    points: list[ReliabilityPoint] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        points.append(
            ReliabilityPoint(
                forecast_mean=float(p_sorted[start:end].mean()),
                observed_rate=float(y_sorted[start:end].mean()),
                fitted_rate=float(fitted[start:end].mean()),
                count=int(end - start),
            )
        )
    return points
```

The Model Lab renders all three one-vs-rest curves, the raw probability histogram, and
sample counts. A class-prior model may look calibrated but has no useful sharpness; the
histogram makes that visible.

## 6. Rolling-origin evaluator and immutable rows (R03, R18, R20)

The evaluator creates a new model instance per fold. No candidate may reuse state fitted
through a later cutoff.

```python
# pitch_oracle_core/evaluation/rolling_origin.py
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from pitch_oracle_core.evaluation.scores import outcome_index
from pitch_oracle_core.models.protocol import FixtureFeatures, ScoreModel


@dataclass(frozen=True)
class EvaluationFold:
    fold_id: str
    train_cutoff_utc: datetime
    test_end_utc: datetime


def evaluate_candidate(
    model_factory: Callable[[], ScoreModel],
    matches: pd.DataFrame,
    folds: Iterable[EvaluationFold],
    feature_builder: Callable[[pd.Series, datetime], FixtureFeatures],
) -> pd.DataFrame:
    frame = matches.copy()
    frame["kickoff_utc"] = pd.to_datetime(frame.kickoff_utc, utc=True)
    rows: list[dict[str, object]] = []

    for fold in folds:
        cutoff = pd.Timestamp(fold.train_cutoff_utc)
        end = pd.Timestamp(fold.test_end_utc)
        train = frame.loc[frame.kickoff_utc < cutoff].copy()
        test = frame.loc[
            (frame.kickoff_utc >= cutoff) & (frame.kickoff_utc < end)
        ].sort_values(["kickoff_utc", "fixture_id"])
        if test.empty:
            continue

        model = model_factory()
        model.spec.validate()
        model.fit(train, cutoff_utc=cutoff.to_pydatetime())

        for _, fixture in test.iterrows():
            features = feature_builder(fixture, cutoff.to_pydatetime())
            if pd.Timestamp(features.kickoff_utc) != fixture.kickoff_utc:
                raise ValueError("feature fixture kickoff does not match evaluation row")
            grid = model.predict_grid(features)
            probability = grid.normalized_one_x_two()
            actual = int(
                outcome_index(
                    np.array([fixture.home_goals]),
                    np.array([fixture.away_goals]),
                )[0]
            )
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "fixture_id": fixture.fixture_id,
                    "kickoff_utc": fixture.kickoff_utc,
                    "issued_at": cutoff,
                    "model_id": model.spec.model_id,
                    "model_family": model.spec.family,
                    "forecast_track": model.spec.track.value,
                    "p_home": probability[0],
                    "p_draw": probability[1],
                    "p_away": probability[2],
                    "tail_mass": grid.tail_mass,
                    "actual_outcome": actual,
                    "actual_home_goals": int(fixture.home_goals),
                    "actual_away_goals": int(fixture.away_goals),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty and result.duplicated(["model_id", "fixture_id"]).any():
        raise ValueError("a fixture was evaluated more than once for a model")
    return result
```

In a production backtest, `feature_builder` must query the feature ledger as of the
fixture's actual issue time, not simply the fold cutoff. The simplified interface above
makes the leakage boundary visible; the final implementation should accept an
`issued_at` schedule such as seven days/24 hours/confirmed lineup.

Persist rows before aggregation:

```text
evaluation_predictions/{experiment_id}/{model_id}.parquet
evaluation_grids/{experiment_id}/{model_id}/{fixture_id}.npz
evaluation_reports/{experiment_id}.json
```

## 7. Paired uncertainty and data-snooping control (R18, R19)

Compare per-fixture losses. Aggregate metrics without pairing waste information and can
mask that candidates were evaluated on different coverage.

```python
# pitch_oracle_core/evaluation/paired_tests.py
from __future__ import annotations

import numpy as np


def circular_block_indices(
    observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if observations < 2 or not 1 <= block_length <= observations:
        raise ValueError("invalid block dimensions")
    blocks = int(np.ceil(observations / block_length))
    starts = rng.integers(0, observations, size=blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % observations).ravel()[:observations]


def paired_block_interval(
    benchmark_loss: np.ndarray,
    candidate_loss: np.ndarray,
    *,
    block_length: int,
    repetitions: int = 5_000,
    seed: int = 20260810,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    benchmark = np.asarray(benchmark_loss, dtype=float)
    candidate = np.asarray(candidate_loss, dtype=float)
    if benchmark.shape != candidate.shape or benchmark.ndim != 1:
        raise ValueError("paired losses must be equal-length vectors")
    # Positive means the candidate has lower/better loss.
    advantage = benchmark - candidate
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample = circular_block_indices(len(advantage), block_length, rng)
        draws[index] = advantage[sample].mean()
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(advantage.mean()), float(lower), float(upper)


def white_style_reality_check(
    loss_advantages: np.ndarray,
    *,
    block_length: int,
    repetitions: int = 5_000,
    seed: int = 20260810,
) -> tuple[float, float]:
    """Approximate White-style test for the best of many candidates.

    `loss_advantages[t, j]` is benchmark loss minus candidate-j loss at time t,
    so positive values favor the candidate. Rows should be ordered matchweeks or
    another dependency-preserving period, not arbitrarily shuffled fixtures.
    """
    values = np.asarray(loss_advantages, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("advantages must have shape (time, candidates)")
    n = values.shape[0]
    observed = float(np.sqrt(n) * np.max(values.mean(axis=0)))
    centered = values - values.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    null_statistics = np.empty(repetitions, dtype=float)
    for draw in range(repetitions):
        sample = circular_block_indices(n, block_length, rng)
        null_statistics[draw] = np.sqrt(n) * np.max(centered[sample].mean(axis=0))
    p_value = (1 + np.count_nonzero(null_statistics >= observed)) / (repetitions + 1)
    return observed, float(p_value)
```

This is a practical White-style bootstrap, not a drop-in implementation of every
regularity correction in White or Hansen. Validate it against a statistical reference
implementation before using p-values as a formal release gate. Even then, practical
effect size and untouched forward performance remain mandatory.

## 8. Multi-method de-vig engine (R14)

Raw inverse decimal odds sum above one. Every conversion stores the original overround
and the selected method.

```python
# pitch_oracle_core/markets/devig.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import numpy as np


class DevigMethod(StrEnum):
    MULTIPLICATIVE = "multiplicative"
    POWER = "power"
    SHIN = "shin"


@dataclass(frozen=True)
class FairMarket:
    probabilities: np.ndarray
    method: DevigMethod
    overround: float
    parameter: float | None

    def __post_init__(self) -> None:
        p = np.asarray(self.probabilities, dtype=float)
        if p.ndim != 1 or (p <= 0).any() or not np.isclose(p.sum(), 1.0, atol=1e-10):
            raise ValueError("fair probabilities must be positive and sum to one")
        object.__setattr__(self, "probabilities", p)


def inverse_odds(decimal_odds: np.ndarray) -> np.ndarray:
    odds = np.asarray(decimal_odds, dtype=float)
    if odds.ndim != 1 or len(odds) < 2 or not np.isfinite(odds).all() or (odds <= 1).any():
        raise ValueError("decimal odds must be a vector strictly above one")
    return 1.0 / odds


def _bisect_root(function, low: float, high: float, iterations: int = 200) -> float:
    f_low = function(low)
    f_high = function(high)
    if f_low == 0:
        return low
    if f_high == 0:
        return high
    if f_low * f_high > 0:
        raise ValueError("root is not bracketed")
    for _ in range(iterations):
        middle = (low + high) / 2
        f_middle = function(middle)
        if abs(f_middle) < 1e-13:
            return middle
        if f_low * f_middle <= 0:
            high = middle
            f_high = f_middle
        else:
            low = middle
            f_low = f_middle
    return (low + high) / 2


def devig(decimal_odds: np.ndarray, method: DevigMethod) -> FairMarket:
    inverse = inverse_odds(decimal_odds)
    total = float(inverse.sum())
    overround = total - 1.0
    if overround < 0:
        raise ValueError("underround markets require an explicit policy")

    if method == DevigMethod.MULTIPLICATIVE:
        return FairMarket(inverse / total, method, overround, None)

    if method == DevigMethod.POWER:
        objective = lambda exponent: float(np.power(inverse, exponent).sum() - 1.0)
        exponent = _bisect_root(objective, 1.0, 100.0)
        probability = np.power(inverse, exponent)
        return FairMarket(probability / probability.sum(), method, overround, exponent)

    if method == DevigMethod.SHIN:
        # Shin's z is found so transformed probabilities sum to one.
        def probabilities(z: float) -> np.ndarray:
            numerator = np.sqrt(
                z * z + 4.0 * (1.0 - z) * np.square(inverse) / total
            ) - z
            return numerator / (2.0 * (1.0 - z))

        objective = lambda z: float(probabilities(z).sum() - 1.0)
        z = _bisect_root(objective, 0.0, 1.0 - 1e-12)
        probability = probabilities(z)
        return FairMarket(probability / probability.sum(), method, overround, z)

    raise ValueError(f"unsupported de-vig method: {method}")
```

Test this implementation against `penaltyblog` and published examples before release.
Numerical convergence failure is a data-quality state, not permission to fall back
silently.

## 9. Market benchmark and closing-price audit (R13, R15, R22)

```python
# pitch_oracle_core/markets/benchmark.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import log
import numpy as np

from pitch_oracle_core.markets.devig import DevigMethod, FairMarket, devig


@dataclass(frozen=True)
class OddsSnapshot:
    fixture_id: str
    market: str
    bookmaker: str
    observed_at: datetime
    decimal_odds: tuple[float, ...]
    source_event_id: str


def fair_snapshot(snapshot: OddsSnapshot, method: DevigMethod) -> FairMarket:
    return devig(np.asarray(snapshot.decimal_odds, dtype=float), method)


def log_closing_line_value(taken_decimal: float, closing_decimal: float) -> float:
    if taken_decimal <= 1 or closing_decimal <= 1:
        raise ValueError("decimal prices must exceed one")
    # Positive means the accepted price was longer/better than the close.
    return log(taken_decimal / closing_decimal)


def consensus_probability(
    markets: list[FairMarket],
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    if not markets:
        raise ValueError("at least one fair market is required")
    values = np.vstack([market.probabilities for market in markets])
    if weights is None:
        weights = np.ones(len(values), dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(values),) or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("invalid consensus weights")
    result = np.average(values, axis=0, weights=weights)
    return result / result.sum()
```

Consensus weights are trained on historical forecast score, never selected on the final
test. Report an equal-weight/median consensus beside any learned source weighting.

Market-implied expected goals belong in a separate adapter:

```python
@dataclass(frozen=True)
class MarketImpliedGoals:
    fixture_id: str
    issued_at: datetime
    expected_home: float
    expected_away: float
    source_markets: tuple[str, ...]
    devig_method: str
    solver_error: float
```

The solver should reconcile 1X2 plus totals/handicap prices to a grid and publish its
residual. It must not pretend an exact solution exists for inconsistent books.

## 10. Exact totals settlement (R21)

Expected return for quarter totals is the average return of the two component legs.
This naturally handles full/half win, push, and full/half loss.

```python
# pitch_oracle_core/markets/settlement.py
from __future__ import annotations

from decimal import Decimal
import numpy as np

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid


def split_quarter_line(line: Decimal) -> tuple[Decimal, Decimal]:
    quarter = line * 4
    if quarter != quarter.to_integral_value():
        raise ValueError("line must be on a quarter-goal increment")
    fraction = line % 1
    if fraction in (Decimal("0.25"), Decimal("0.75")):
        return line - Decimal("0.25"), line + Decimal("0.25")
    return line, line


def over_leg_net_return(total_goals: int, line: Decimal, decimal_odds: Decimal) -> Decimal:
    if decimal_odds <= 1:
        raise ValueError("decimal odds must exceed one")
    total = Decimal(total_goals)
    if total > line:
        return decimal_odds - 1
    if total == line:
        return Decimal(0)
    return Decimal(-1)


def over_quarter_net_return(
    total_goals: int,
    line: Decimal,
    decimal_odds: Decimal,
) -> Decimal:
    left, right = split_quarter_line(line)
    return (
        over_leg_net_return(total_goals, left, decimal_odds)
        + over_leg_net_return(total_goals, right, decimal_odds)
    ) / 2


def expected_over_net_return(
    grid: ProbabilityGrid,
    line: Decimal,
    decimal_odds: Decimal,
    *,
    max_tail: float = 1e-8,
) -> float:
    if grid.tail_mass > max_tail:
        raise ValueError("grid tail is too large for point expected return")
    expected = Decimal(0)
    for home_goals in range(grid.mass.shape[0]):
        for away_goals in range(grid.mass.shape[1]):
            probability = Decimal(str(grid.mass[home_goals, away_goals]))
            expected += probability * over_quarter_net_return(
                home_goals + away_goals, line, decimal_odds
            )
    # Tiny tail is omitted and surfaced in the artifact as an error bound.
    return float(expected)
```

Build equivalent Asian-handicap settlement over home-goal minus away-goal. All market
tests should enumerate score pairs and lines from -5.0 through +5.0 in quarter steps.

## 11. Empirical hierarchical priors as a safe first step (R09)

A full Bayesian state-space model can follow. First, replace invented histories with a
transparent partial-pooling prior.

```python
# pitch_oracle_core/models/hierarchical_prior.py
from dataclasses import dataclass


@dataclass(frozen=True)
class StrengthPrior:
    attack_mean: float
    defense_mean: float
    effective_matches: float
    provenance: str


def partial_pool_strength(
    observed_mean: float,
    observed_matches: float,
    prior_mean: float,
    prior_matches: float,
) -> tuple[float, float]:
    if observed_matches < 0 or prior_matches <= 0:
        raise ValueError("invalid effective sample sizes")
    weight = observed_matches / (observed_matches + prior_matches)
    return weight * observed_mean + (1 - weight) * prior_mean, weight


def promoted_team_prior(
    lower_division_attack: float | None,
    lower_division_defense: float | None,
    league_attack_mean: float,
    league_defense_mean: float,
    translation_weight: float,
) -> StrengthPrior:
    if not 0 <= translation_weight <= 1:
        raise ValueError("translation_weight must be in [0, 1]")
    if lower_division_attack is None or lower_division_defense is None:
        return StrengthPrior(
            league_attack_mean,
            league_defense_mean,
            effective_matches=0.0,
            provenance="league_prior_no_lower_division_evidence",
        )
    return StrengthPrior(
        translation_weight * lower_division_attack
        + (1 - translation_weight) * league_attack_mean,
        translation_weight * lower_division_defense
        + (1 - translation_weight) * league_defense_mean,
        effective_matches=translation_weight * 8.0,
        provenance="translated_lower_division_prior",
    )
```

Estimate `translation_weight` across past promoted cohorts in nested folds. Never set it
from the current promoted club's future season.

## 12. Provider-neutral events and capability gates (R23–R25)

```python
# pitch_oracle_core/events/schema.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Mapping


class ActionType(StrEnum):
    PASS = "pass"
    CARRY = "carry"
    DRIBBLE = "dribble"
    SHOT = "shot"
    DUEL = "duel"
    RECOVERY = "recovery"
    FOUL = "foul"
    KEEPER = "keeper"
    OTHER = "other"


@dataclass(frozen=True)
class CanonicalAction:
    fixture_id: str
    action_id: str
    sequence_index: int
    period: int
    seconds_in_period: float
    team_id: str
    player_id: str | None
    action_type: ActionType
    start_x: float | None       # normalized 0..105, team attacks left-to-right
    start_y: float | None       # normalized 0..68
    end_x: float | None
    end_y: float | None
    successful: bool | None
    body_part: str | None
    set_piece: str | None
    observed_at: datetime
    provider: str
    provider_event_id: str
    provider_schema_version: str
    raw_attributes: Mapping[str, object]

    def validate(self) -> None:
        if self.sequence_index < 0 or self.period < 1 or self.seconds_in_period < 0:
            raise ValueError("invalid event ordering fields")
        for x in (self.start_x, self.end_x):
            if x is not None and not 0 <= x <= 105:
                raise ValueError("x coordinate outside canonical pitch")
        for y in (self.start_y, self.end_y):
            if y is not None and not 0 <= y <= 68:
                raise ValueError("y coordinate outside canonical pitch")
```

Capability states distinguish unavailable data from observed zero:

```python
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class CapabilityStatus(StrEnum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class ProviderCapability:
    competition_id: str
    edition_id: str
    capability: str
    status: CapabilityStatus
    coverage_fraction: float
    observed_at: datetime
    provider: str
    reason: str | None
```

The model registry refuses an event candidate when its required capability is not
available for the evaluation row. Coverage-matched comparisons report both the common
subset and the universal fallback path.

## 13. Player and lineup strength bridge (R12, R24)

```python
# pitch_oracle_core/players/lineup_strength.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PlayerStrength:
    player_id: str
    attack_per_90: float
    defense_per_90: float
    effective_minutes: float
    estimated_at: datetime
    model_id: str


@dataclass(frozen=True)
class LineupMember:
    player_id: str
    expected_minutes: float
    availability_probability: float


def shrink_player(value: float, minutes: float, prior_minutes: float = 900.0) -> float:
    if minutes < 0 or prior_minutes <= 0:
        raise ValueError("invalid minutes")
    return value * minutes / (minutes + prior_minutes)


def lineup_delta(
    members: list[LineupMember],
    strengths: dict[str, PlayerStrength],
    replacement_attack_per_90: float,
    replacement_defense_per_90: float,
) -> tuple[float, float, float]:
    attack = 0.0
    defense = 0.0
    covered_minutes = 0.0
    for member in members:
        if not 0 <= member.availability_probability <= 1:
            raise ValueError("availability probability outside [0, 1]")
        expected = member.expected_minutes * member.availability_probability
        if not 0 <= expected <= 90:
            raise ValueError("expected player minutes outside [0, 90]")
        strength = strengths.get(member.player_id)
        if strength is None:
            continue
        attack += expected / 90 * shrink_player(
            strength.attack_per_90, strength.effective_minutes
        )
        defense += expected / 90 * shrink_player(
            strength.defense_per_90, strength.effective_minutes
        )
        covered_minutes += expected
    # Eleven player-slots × 90 minutes. Missing mass gets replacement strength.
    replacement_minutes = max(0.0, 990.0 - covered_minutes)
    attack += replacement_minutes / 90 * replacement_attack_per_90
    defense += replacement_minutes / 90 * replacement_defense_per_90
    return attack, defense, covered_minutes / 990.0
```

Player contributions must be estimated strictly before the target fixture. Confirmed
lineups generate a new issue, while the earlier pre-lineup forecast remains in the
ledger.

## 14. Versioned experiment specifications (R19, R20)

Do not let an analyst change the metric or threshold after seeing the final results.

```yaml
# experiments/2026-09-belgium-distributions.yml
experiment_id: 2026-09-belgium-distributions-v1
hypothesis: >
  A recency-weighted dependent-score model improves probabilistic Belgium forecasts
  over the current independent Poisson baseline, especially draw calibration.
data_cutoff_created_at: 2026-09-01T00:00:00Z
leagues:
  - belgian-pro-league
tracks:
  - independent
candidates:
  - independent-poisson-v2
  - dixon-coles-v1
  - bivariate-poisson-v1
  - diagonal-inflated-bivariate-v1
primary_metric: multiclass_log_loss
secondary_metrics:
  - multiclass_brier
  - ranked_probability_score
  - scoreline_ignorance
mandatory_cohorts:
  - draw
  - early_season
  - promoted_or_cold_start
  - regular_phase
  - playoff_phase
tuning_window_end: 2025-06-30T23:59:59Z
forward_test_start: 2025-07-01T00:00:00Z
forward_test_end: 2026-06-30T23:59:59Z
paired_block_unit: matchweek
family_test: white_style_reality_check
max_tail_mass: 1.0e-8
promotion_minimum:
  log_loss_relative_improvement: 0.005
  calibration_regression_allowed: false
  fallback_failure_rate_max: 0.001
```

Hash the experiment spec into every output artifact. A revised candidate list creates a
new experiment version and a new untouched forward range.

## 15. Research Lab UI

The UI should make tradeoffs legible, not crown a model with a green trophy because its
accuracy is 0.4 percentage points higher.

```python
# pitch_oracle_core/ui/pages/research_lab.py
import pandas as pd
import streamlit as st


def render_research_lab(
    experiments: pd.DataFrame,
    metric_rows: pd.DataFrame,
    calibration_rows: pd.DataFrame,
) -> None:
    st.title("Forecast Research Lab")
    st.caption(
        "All results are rolling-origin forecasts. Betting simulation is a separate "
        "decision-policy evaluation and does not prove future returns."
    )
    experiment_id = st.selectbox(
        "Frozen experiment",
        experiments.experiment_id.tolist(),
    )
    experiment = experiments.loc[
        experiments.experiment_id == experiment_id
    ].iloc[0]
    st.write(
        {
            "hypothesis": experiment.hypothesis,
            "created": experiment.created_at,
            "forward test": experiment.forward_test_range,
            "family test": experiment.family_test,
            "status": experiment.status,
        }
    )

    metrics = metric_rows.loc[metric_rows.experiment_id == experiment_id]
    st.subheader("Probability scorecard")
    st.dataframe(
        metrics[
            [
                "model_id", "track", "fixtures", "log_loss", "brier", "rps",
                "draw_reliability_gap", "market_log_loss_delta", "fit_failures",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Calibration and sharpness")
    st.caption(
        "Choose a class and cohort. The probability histogram is shown with the "
        "reliability curve so a low-information prior cannot look impressive by itself."
    )
    # Use the shared Plotly component described in 05-ui-implementation.md.
    render_calibration_triptych(
        calibration_rows.loc[calibration_rows.experiment_id == experiment_id]
    )

    st.subheader("Decision record")
    st.markdown(experiment.decision_markdown)
```

Additional tabs:

- residual lab: dispersion, zeros, diagonal, scoreline heatmap residuals;
- paired deltas: per-fold change and block-bootstrap interval;
- cohorts: league/edition/phase/cold-start/horizon/data-quality;
- market benchmark: independent, market-aware, and consensus side by side;
- operations: fit time, inference time, artifact size, convergence and fallback rate;
- experiment graveyard: rejected candidates and why.

## 16. Tests that prevent persuasive nonsense

```python
# tests/test_research_contracts.py
from decimal import Decimal
import numpy as np
import pytest

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid
from pitch_oracle_core.evaluation.scores import (
    multiclass_brier,
    multiclass_log_loss,
    ranked_probability_score,
)
from pitch_oracle_core.markets.devig import DevigMethod, devig
from pitch_oracle_core.markets.settlement import over_quarter_net_return


def test_probability_grid_preserves_explicit_tail():
    mass = np.array([[0.25, 0.20], [0.25, 0.20]])
    grid = ProbabilityGrid(mass, 0.10, 1, 1)
    assert grid.represented_mass == pytest.approx(0.90)
    with pytest.raises(ValueError, match="tail"):
        grid.normalized_one_x_two()


def test_perfect_forecast_beats_uniform_on_all_primary_scores():
    y = np.array([0, 1, 2])
    perfect = np.eye(3) * 0.999998 + (1 - np.eye(3)) * 0.000001
    uniform = np.full((3, 3), 1 / 3)
    assert multiclass_log_loss(y, perfect) < multiclass_log_loss(y, uniform)
    assert multiclass_brier(y, perfect) < multiclass_brier(y, uniform)
    assert ranked_probability_score(y, perfect) < ranked_probability_score(y, uniform)


@pytest.mark.parametrize(
    "method",
    [DevigMethod.MULTIPLICATIVE, DevigMethod.POWER, DevigMethod.SHIN],
)
def test_devig_probabilities_sum_to_one(method):
    fair = devig(np.array([2.10, 3.40, 3.60]), method)
    assert fair.probabilities.sum() == pytest.approx(1.0)
    assert (fair.probabilities > 0).all()
    assert fair.overround > 0


@pytest.mark.parametrize(
    ("total", "line", "expected"),
    [
        (3, "2.25", "1.00"),      # both over legs win at evens
        (2, "2.25", "-0.50"),     # push at 2.0, lose at 2.5
        (2, "1.75", "0.50"),      # win at 1.5, push at 2.0
        (1, "1.75", "-1.00"),
    ],
)
def test_quarter_total_settlement(total, line, expected):
    returned = over_quarter_net_return(total, Decimal(line), Decimal("2.00"))
    assert returned == Decimal(expected)
```

Add these non-negotiable tests:

- shuffled future results change no pre-kickoff feature row;
- a fixture's `issued_at` is before kickoff and never mutates after result ingestion;
- independent model manifests contain no odds capability or odds-derived column;
- every candidate is evaluated on the same fixture IDs or reports coverage-matched and
  universal-fallback scores separately;
- modal exact score and most likely 1X2 outcome are computed/labeled independently;
- score-grid markets reconcile within tolerance;
- de-vig golden examples match a trusted numerical implementation;
- bivariate/Dixon-Coles invalid parameter regions fail rather than produce negative mass;
- model search logs every candidate and threshold;
- forward-test partitions cannot be loaded by tuning jobs;
- Belgium regular/playoff phase tags are correct at issue time;
- source odds observed after issue time cannot enter that forecast or simulated decision.

## 17. Ordered delivery

### Research release A — honest comparison foundation

Implement R01, R02, R16–R20:

- common grid/model protocol;
- immutable rolling-origin rows;
- log/Brier/RPS/scoreline panel;
- CORP-style reliability and sharpness;
- paired block intervals and experiment specs;
- Research Lab scorecard.

No new champion is required for this release. The outcome is an evaluator that can say
“none of the challengers improved enough.”

### Research release B — Belgium distribution tournament

Implement R03–R06 and R10:

- tuned recency;
- Dixon-Coles;
- bivariate and diagonal-inflated challenger;
- NB/CMP only when diagnostics justify them;
- Elo/Pi/Glicko calibrated baselines.

Freeze the candidate list and forward range before viewing results.

### Research release C — market benchmark

Implement R13–R15 and R21–R22:

- independent/market-aware identities;
- multiplicative/power/Shin conversion;
- source and consensus market baselines;
- exact totals/Asian settlement;
- closing-price/CLV ledger;
- realistic costs and threshold family tests.

This release may conclude there is no credible executable edge. That is a valid and
useful result.

### Research release D — richer data

Implement R12 and R23–R25 only where coverage exists:

- canonical shot/action data;
- calibrated provider-neutral xG;
- xT/VAEP lagged team/player summaries;
- expected-lineup strength;
- tracking lab behind a non-production capability.

### Research release E — uncertainty-aware simulation

Implement R08, R09, and R26:

- dynamic/hierarchical strengths;
- posterior/parameter draws through match and season forecasts;
- uncertainty-aware zero-default portfolio simulations;
- long shadow evaluation before any market-facing rollout.

## 18. Belgium acceptance checklist

Before the Belgium example can promote a researched model:

- canonical entity coverage is 100% for active teams;
- phase/rules fixtures pass regular-season and playoff tests;
- current generic/duplicate vector counts are published as the baseline;
- class-prior, independent Poisson, Elo, and de-vigged market (where available) are in
  the same frozen evaluation table;
- at least three full historical edition boundaries appear in tuning/evaluation, or the
  insufficient-history limitation is explicit;
- draw calibration is shown with uncertainty, not only recall;
- every count model has finite, non-negative, reconciled grid mass;
- fit/convergence fallback frequency is below the frozen threshold;
- no provider/event/player data was retrospectively joined;
- final forward rows were not used to choose distribution, decay, calibration, edge,
  or staking thresholds;
- the thin `belgium-soccer` repository only consumes the resulting versioned artifacts.

## 19. Final engineering stance

The research program should make Pitch Oracle more adventurous and more skeptical at
the same time. It can support bivariate counts, dynamic Bayesian strengths, player
ratings, event value, pitch control, market comparison, and portfolio simulation. But
every one of those ideas enters through the same contracts, faces the same chronological
rows, and can lose to a simple baseline.

That is the durable feature: not any one model, but a platform that can discover which
ideas actually transfer to Belgium and the rest of European soccer without sacrificing
coherence, provenance, or honesty.
