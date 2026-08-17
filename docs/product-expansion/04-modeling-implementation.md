# Modeling implementation

The objective is not to maximize model count. It is to produce a coherent forecast
distribution, quantify uncertainty, and deploy complexity only when it beats simple
baselines out of time. The existing walk-forward Poisson remains a valuable baseline
and fallback.

## Forecast contract

Every candidate must emit the same object. All derived markets and simulations use
the score matrix, preventing contradictory probabilities.

```python
# pitch_oracle_core/domain/forecasts.py
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass(frozen=True)
class MatchForecast:
    fixture_id: str
    issued_at: datetime
    model_id: str
    score_matrix: np.ndarray
    home_history_n: int
    away_history_n: int
    prior_weight: float
    entity_resolution_status: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.score_matrix, dtype=float)
        if matrix.ndim != 2 or not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("score_matrix must be finite, non-negative, and 2D")
        if not np.isclose(matrix.sum(), 1.0, atol=1e-8):
            raise ValueError("score_matrix must sum to one")

    @property
    def one_x_two(self) -> tuple[float, float, float]:
        home = float(np.tril(self.score_matrix, k=-1).sum())
        draw = float(np.trace(self.score_matrix))
        away = float(np.triu(self.score_matrix, k=1).sum())
        return home, draw, away

    @property
    def expected_goals(self) -> tuple[float, float]:
        home_goals = np.arange(self.score_matrix.shape[0])[:, None]
        away_goals = np.arange(self.score_matrix.shape[1])[None, :]
        return (
            float((self.score_matrix * home_goals).sum()),
            float((self.score_matrix * away_goals).sum()),
        )

    def total_over(self, line: float) -> float:
        if line < 0 or line % 1 != 0.5:
            raise ValueError("Only non-negative half-goal lines are supported")
        home_goals = np.arange(self.score_matrix.shape[0])[:, None]
        away_goals = np.arange(self.score_matrix.shape[1])[None, :]
        return float(self.score_matrix[(home_goals + away_goals) > line].sum())

    @property
    def btts(self) -> float:
        return float(self.score_matrix[1:, 1:].sum())
```

Persist matrices as float32 with a declared `max_goals` and `tail_mass`. Do not round
before deriving markets.

## 1. Dynamic Elo baseline (F19, F21, F40, F42)

Elo is fast, interpretable, and naturally point-in-time. Regress ratings toward the
edition prior between seasons; promoted clubs can use a lower-division or conservative
country prior without inventing fake match history.

```python
# pitch_oracle_core/models/elo.py
from dataclasses import dataclass, field
from math import log10


@dataclass
class EloModel:
    base_rating: float = 1500.0
    k_factor: float = 24.0
    home_advantage: float = 55.0
    season_regression: float = 0.25
    ratings: dict[str, float] = field(default_factory=dict)
    matches: dict[str, int] = field(default_factory=dict)

    def rating(self, team_id: str, prior: float | None = None) -> float:
        return self.ratings.get(team_id, prior or self.base_rating)

    @staticmethod
    def expected(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def score(home_goals: int, away_goals: int) -> float:
        return 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0

    @staticmethod
    def goal_multiplier(goal_difference: int) -> float:
        difference = abs(goal_difference)
        if difference <= 1:
            return 1.0
        return 1.0 + 0.5 * log10(float(difference))

    def update(
        self,
        home_team_id: str,
        away_team_id: str,
        home_goals: int,
        away_goals: int,
        *,
        home_prior: float | None = None,
        away_prior: float | None = None,
    ) -> tuple[float, float]:
        home = self.rating(home_team_id, home_prior)
        away = self.rating(away_team_id, away_prior)
        expected_home = self.expected(home + self.home_advantage, away)
        actual_home = self.score(home_goals, away_goals)
        change = (
            self.k_factor
            * self.goal_multiplier(home_goals - away_goals)
            * (actual_home - expected_home)
        )
        self.ratings[home_team_id] = home + change
        self.ratings[away_team_id] = away - change
        self.matches[home_team_id] = self.matches.get(home_team_id, 0) + 1
        self.matches[away_team_id] = self.matches.get(away_team_id, 0) + 1
        return self.ratings[home_team_id], self.ratings[away_team_id]

    def regress_for_new_edition(self) -> None:
        for team_id, rating in self.ratings.items():
            self.ratings[team_id] = (
                (1.0 - self.season_regression) * rating
                + self.season_regression * self.base_rating
            )
```

Use Elo as:

- a feature and baseline, not an uncalibrated final 1X2 probability;
- opponent strength for adjusted form;
- fixture-difficulty and power-ranking input;
- promoted-team prior provenance;
- a time-travel state saved after every completed fixture.

## 2. Time-decayed Dixon-Coles goals model (F02, F03, F09, F39, F43)

This is a practical improvement over independent full-history Poisson because it
models low-score correlation and lets old matches decay.

```python
# pitch_oracle_core/models/dixon_coles.py
from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def poisson_log_pmf(goals: int, rate: float) -> float:
    return goals * log(rate) - rate - lgamma(goals + 1.0)


def dc_tau(home_goals: int, away_goals: int, home_rate: float, away_rate: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - home_rate * away_rate * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + home_rate * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + away_rate * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


@dataclass(frozen=True)
class DixonColesFit:
    team_ids: tuple[str, ...]
    attack: np.ndarray
    defense: np.ndarray
    intercept: float
    home_advantage: float
    rho: float
    fitted_at: pd.Timestamp
    half_life_days: float


class DixonColesModel:
    def __init__(self, half_life_days: float = 365.0, l2: float = 0.02) -> None:
        self.half_life_days = half_life_days
        self.l2 = l2
        self.fit_: DixonColesFit | None = None

    def fit(self, matches: pd.DataFrame, as_of: object | None = None) -> "DixonColesModel":
        required = {
            "kickoff_utc", "home_team_id", "away_team_id", "home_goals", "away_goals"
        }
        missing = required.difference(matches.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        frame = matches.dropna(subset=list(required)).copy()
        frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True)
        cutoff = pd.Timestamp(as_of or frame["kickoff_utc"].max())
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        frame = frame.loc[frame["kickoff_utc"] < cutoff]
        teams = tuple(sorted(set(frame.home_team_id) | set(frame.away_team_id)))
        if len(teams) < 2 or len(frame) < len(teams):
            raise ValueError("Insufficient history for Dixon-Coles fit")
        index = {team_id: position for position, team_id in enumerate(teams)}
        home_index = frame.home_team_id.map(index).to_numpy()
        away_index = frame.away_team_id.map(index).to_numpy()
        home_goals = frame.home_goals.astype(int).to_numpy()
        away_goals = frame.away_goals.astype(int).to_numpy()
        age_days = (cutoff - frame.kickoff_utc).dt.total_seconds().to_numpy() / 86_400
        weights = np.exp(-np.log(2.0) * age_days / self.half_life_days)
        size = len(teams)

        def unpack(parameters: np.ndarray):
            attack = parameters[:size]
            defense = parameters[size:2 * size]
            # Center both vectors to make the representation identifiable.
            attack = attack - attack.mean()
            defense = defense - defense.mean()
            intercept, home_advantage, rho = parameters[-3:]
            return attack, defense, intercept, home_advantage, rho

        def objective(parameters: np.ndarray) -> float:
            attack, defense, intercept, home_advantage, rho = unpack(parameters)
            home_rate = np.exp(
                intercept + home_advantage + attack[home_index] - defense[away_index]
            )
            away_rate = np.exp(intercept + attack[away_index] - defense[home_index])
            log_likelihood = np.empty(len(frame), dtype=float)
            for row in range(len(frame)):
                tau = dc_tau(
                    home_goals[row], away_goals[row], home_rate[row], away_rate[row], rho
                )
                if tau <= 0:
                    return 1e12
                log_likelihood[row] = (
                    log(tau)
                    + poisson_log_pmf(home_goals[row], home_rate[row])
                    + poisson_log_pmf(away_goals[row], away_rate[row])
                )
            penalty = self.l2 * (
                np.square(attack).sum() + np.square(defense).sum()
            )
            return float(-(weights * log_likelihood).sum() + penalty)

        initial = np.zeros(2 * size + 3, dtype=float)
        initial[-3] = log(max((home_goals.sum() + away_goals.sum()) / (2 * len(frame)), 0.2))
        initial[-2] = 0.15
        bounds = [(None, None)] * (2 * size + 2) + [(-0.2, 0.2)]
        result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Dixon-Coles fit failed: {result.message}")
        attack, defense, intercept, home_advantage, rho = unpack(result.x)
        self.fit_ = DixonColesFit(
            teams, attack, defense, float(intercept), float(home_advantage),
            float(rho), cutoff, self.half_life_days,
        )
        return self

    def expected_goals(self, home_team_id: str, away_team_id: str) -> tuple[float, float]:
        if self.fit_ is None:
            raise RuntimeError("Model is not fitted")
        index = {team_id: position for position, team_id in enumerate(self.fit_.team_ids)}
        if home_team_id not in index or away_team_id not in index:
            raise KeyError("Unknown team; resolve an explicit promoted-team prior first")
        home = index[home_team_id]
        away = index[away_team_id]
        home_rate = np.exp(
            self.fit_.intercept + self.fit_.home_advantage
            + self.fit_.attack[home] - self.fit_.defense[away]
        )
        away_rate = np.exp(
            self.fit_.intercept + self.fit_.attack[away] - self.fit_.defense[home]
        )
        return float(home_rate), float(away_rate)

    def score_matrix(
        self, home_team_id: str, away_team_id: str, max_goals: int = 10
    ) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("Model is not fitted")
        home_rate, away_rate = self.expected_goals(home_team_id, away_team_id)
        matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                probability = np.exp(
                    poisson_log_pmf(home_goals, home_rate)
                    + poisson_log_pmf(away_goals, away_rate)
                )
                probability *= dc_tau(
                    home_goals, away_goals, home_rate, away_rate, self.fit_.rho
                )
                matrix[home_goals, away_goals] = probability
        if (matrix < 0).any() or matrix.sum() <= 0:
            raise ValueError("Invalid Dixon-Coles score distribution")
        return matrix / matrix.sum()
```

Production additions:

- fit one global edition model plus a pooled multi-league prior for new clubs;
- tune half-life and L2 only inside rolling-origin folds;
- save parameter covariance or bootstrap fits for intervals;
- record truncated tail mass before renormalization;
- recalibrate 1X2 only by adjusting the joint matrix, never by independently changing
  1X2 columns while leaving score markets untouched.

## 3. Coherent market derivation (F03, F43)

```python
def markets_from_score_matrix(matrix: np.ndarray) -> dict[str, float | str]:
    matrix = np.asarray(matrix, dtype=float)
    matrix = matrix / matrix.sum()
    home_goals = np.arange(matrix.shape[0])[:, None]
    away_goals = np.arange(matrix.shape[1])[None, :]
    total = home_goals + away_goals
    home_win = float(matrix[home_goals > away_goals].sum())
    draw = float(matrix[home_goals == away_goals].sum())
    away_win = float(matrix[home_goals < away_goals].sum())
    mode = np.unravel_index(int(matrix.argmax()), matrix.shape)
    output: dict[str, float | str] = {
        "p_home": home_win,
        "p_draw": draw,
        "p_away": away_win,
        "p_btts_yes": float(matrix[1:, 1:].sum()),
        "p_home_clean_sheet": float(matrix[:, 0].sum()),
        "p_away_clean_sheet": float(matrix[0, :].sum()),
        "p_home_win_to_nil": float(matrix[1:, 0].sum()),
        "p_away_win_to_nil": float(matrix[0, 1:].sum()),
        "most_likely_score": f"{mode[0]}-{mode[1]}",
    }
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        suffix = str(line).replace(".", "_")
        output[f"p_over_{suffix}"] = float(matrix[total > line].sum())
        output[f"p_under_{suffix}"] = float(matrix[total < line].sum())
    return output
```

## 4. Out-of-fold probability stacking (F36, F41)

Only out-of-fold component predictions may train the stacker. The test period must
remain untouched until model selection.

```python
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityStacker:
    estimator: LogisticRegression | None = None
    component_names: tuple[str, ...] = ()

    @staticmethod
    def _matrix(predictions: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
        names = tuple(sorted(predictions))
        arrays = []
        row_count = None
        for name in names:
            probability = np.asarray(predictions[name], dtype=float)
            if probability.ndim != 2 or probability.shape[1] != 3:
                raise ValueError(f"{name} must have shape (rows, 3)")
            probability = np.clip(probability, 1e-8, 1.0)
            probability /= probability.sum(axis=1, keepdims=True)
            row_count = row_count or len(probability)
            if len(probability) != row_count:
                raise ValueError("Component row counts disagree")
            arrays.append(np.log(probability))
        return np.hstack(arrays), names

    def fit(self, oof_predictions: dict[str, np.ndarray], y: np.ndarray):
        matrix, names = self._matrix(oof_predictions)
        self.estimator = LogisticRegression(
            penalty="l2", C=0.2, solver="lbfgs", max_iter=2_000,
        ).fit(matrix, y)
        self.component_names = names
        return self

    def predict_proba(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Stacker is not fitted")
        matrix, names = self._matrix(predictions)
        if names != self.component_names:
            raise ValueError("Stacker component contract changed")
        probability = self.estimator.predict_proba(matrix)
        return probability / probability.sum(axis=1, keepdims=True)
```

If the stacker wins 1X2 metrics but has no joint goals distribution, project its 1X2
targets back onto the base score matrix with iterative proportional fitting. Do not
publish unreconciled score and outcome forecasts.

## 5. Bootstrap uncertainty (F07, F29, F45)

Use parameter/model bootstrap draws, not the spread of three arbitrary model types as
an uncertainty estimate.

```python
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ForecastInterval:
    median: np.ndarray
    lower_50: np.ndarray
    upper_50: np.ndarray
    lower_80: np.ndarray
    upper_80: np.ndarray


def probability_intervals(draws: np.ndarray) -> ForecastInterval:
    """draws shape: (bootstrap_draw, fixture, outcome)."""
    values = np.asarray(draws, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("Expected draws with shape (draw, fixture, 3)")
    values /= values.sum(axis=2, keepdims=True)
    return ForecastInterval(
        median=np.quantile(values, 0.50, axis=0),
        lower_50=np.quantile(values, 0.25, axis=0),
        upper_50=np.quantile(values, 0.75, axis=0),
        lower_80=np.quantile(values, 0.10, axis=0),
        upper_80=np.quantile(values, 0.90, axis=0),
    )


def interval_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    targets = np.eye(3)[np.asarray(y_true, dtype=int)]
    return ((targets >= lower) & (targets <= upper)).mean(axis=0)
```

The UI stability label can use the bootstrap proportion agreeing on the leading
outcome. A 51% leader in 52% of draws is fragile even if its point estimate looks
decisive.

## 6. Rule-aware season simulation (F27–F31)

Simulation consumes fixture score matrices and a competition rule adapter. Special
formats remain data-driven because the adapter is selected by `rules_version`, not
league name.

```python
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np


@dataclass(frozen=True)
class SimulationFixture:
    fixture_id: str
    home_team_id: str
    away_team_id: str
    score_matrix: np.ndarray


class CompetitionRuleAdapter:
    """Implemented by regular, split-pool, or playoff rule engines."""
    def initial_state(self, completed_matches): ...
    def apply_score(self, state, fixture, home_goals: int, away_goals: int): ...
    def next_fixtures(self, state) -> list[SimulationFixture]: ...
    def is_complete(self, state) -> bool: ...
    def ranked_teams(self, state) -> list[str]: ...
    def outcome_labels(self, state) -> dict[str, set[str]]: ...


def sample_score(matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flattened = np.asarray(matrix, dtype=float).ravel()
    flattened /= flattened.sum()
    index = int(rng.choice(len(flattened), p=flattened))
    return np.unravel_index(index, matrix.shape)


def simulate_season(
    rules: CompetitionRuleAdapter,
    completed_matches,
    initial_fixtures: list[SimulationFixture],
    *,
    simulations: int = 20_000,
    seed: int = 20260810,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    positions: dict[str, Counter[int]] = defaultdict(Counter)
    outcomes: dict[str, Counter[str]] = defaultdict(Counter)

    for _ in range(simulations):
        state = rules.initial_state(completed_matches)
        queue = list(initial_fixtures)
        while not rules.is_complete(state):
            if not queue:
                queue.extend(rules.next_fixtures(state))
                if not queue:
                    raise RuntimeError("Rules produced no fixtures for incomplete season")
            fixture = queue.pop(0)
            home_goals, away_goals = sample_score(fixture.score_matrix, rng)
            rules.apply_score(state, fixture, home_goals, away_goals)
        ranking = rules.ranked_teams(state)
        labels = rules.outcome_labels(state)
        for position, team_id in enumerate(ranking, start=1):
            positions[team_id][position] += 1
        for label, members in labels.items():
            for team_id in members:
                outcomes[team_id][label] += 1

    return {
        "simulations": simulations,
        "position_probabilities": {
            team_id: {position: count / simulations for position, count in counts.items()}
            for team_id, counts in positions.items()
        },
        "outcome_probabilities": {
            team_id: {label: count / simulations for label, count in counts.items()}
            for team_id, counts in outcomes.items()
        },
    }
```

For F29, rerun or reweight common simulation draws conditional on the focal fixture's
home/draw/away result and measure the change in named outcome probabilities. Common
random numbers reduce visual jitter.

## 7. Contextual evaluation and release gate (F36–F38, F44–F45)

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def multiclass_brier(y_true: np.ndarray, probability: np.ndarray) -> float:
    targets = np.eye(3)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.square(probability - targets).sum(axis=1)))


def calibration_table(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    predicted = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    correct = predicted == y_true
    bucket = pd.cut(confidence, np.linspace(0, 1, bins + 1), include_lowest=True)
    return pd.DataFrame({"bucket": bucket, "confidence": confidence, "correct": correct}).groupby(
        "bucket", observed=False
    ).agg(n=("correct", "size"), mean_confidence=("confidence", "mean"), accuracy=("correct", "mean")).reset_index()


def paired_bootstrap_delta(
    y_true: np.ndarray,
    champion: np.ndarray,
    challenger: np.ndarray,
    *,
    draws: int = 2_000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    deltas = []
    rows = len(y_true)
    for _ in range(draws):
        sample = rng.integers(0, rows, rows)
        delta = (
            log_loss(y_true[sample], challenger[sample], labels=[0, 1, 2])
            - log_loss(y_true[sample], champion[sample], labels=[0, 1, 2])
        )
        deltas.append(delta)
    return {
        "median_champion_advantage": float(np.median(deltas)),
        "lower_95": float(np.quantile(deltas, 0.025)),
        "upper_95": float(np.quantile(deltas, 0.975)),
        "probability_champion_better": float(np.mean(np.asarray(deltas) > 0)),
    }


def cohort_metrics(evaluations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, group in evaluations.groupby("cohort", observed=True):
        probability = group[["p_home", "p_draw", "p_away"]].to_numpy()
        y = group["target"].to_numpy(dtype=int)
        rows.append({
            "cohort": cohort,
            "n": len(group),
            "log_loss": log_loss(y, probability, labels=[0, 1, 2]),
            "brier": multiclass_brier(y, probability),
            "draw_recall": float((probability.argmax(axis=1)[y == 1] == 1).mean())
                if (y == 1).any() else np.nan,
        })
    return pd.DataFrame(rows)
```

Required cohorts:

- competition edition and rolling 90/180/365-day windows
- first five matchdays vs established season
- promoted/new/low-history clubs
- home favorite, balanced, away favorite probability bands
- short rest and congestion
- regular vs split/playoff phase
- full vs degraded provider/entity coverage
- probability confidence deciles

Promotion gate:

1. challenger improves rolling-origin log loss and Brier against the production model
   or has a clearly documented Pareto tradeoff;
2. paired bootstrap probability of improvement is at least 90%;
3. no critical cohort exceeds configured degradation bounds;
4. calibration and interval coverage pass;
5. cross-market invariants pass;
6. inference and artifact budgets pass;
7. model card and deterministic reproduction command are present.

## 8. Drift report (F44, F48)

Start with transparent drift metrics before adding a monitoring platform.

```python
def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference[np.isfinite(reference)], np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    reference_share, _ = np.histogram(reference, bins=edges)
    current_share, _ = np.histogram(current, bins=edges)
    reference_share = np.clip(reference_share / reference_share.sum(), 1e-6, 1.0)
    current_share = np.clip(current_share / current_share.sum(), 1e-6, 1.0)
    return float(((current_share - reference_share) * np.log(current_share / reference_share)).sum())


def drift_severity(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "watch"
    return "action_required"
```

Report data drift, prediction drift, coverage drift, and realized performance drift
separately. Retraining should not be automatic merely because one metric moved;
critical entity/time/schema failures block publication, while model drift opens a
challenger evaluation.

## 9. Model registry artifact

```json
{
  "schema_version": 1,
  "production_model_id": "bel.1:dc-elo-stack:2026-08-10",
  "models": [
    {
      "model_id": "bel.1:dc:2026-08-10",
      "family": "dixon_coles",
      "status": "component",
      "trained_through": "2026-08-09T23:59:59Z",
      "feature_set_version": "team-state-v1",
      "entity_registry_version": "bel-teams-v2",
      "rules_version": "bel.1-2026-27-v1",
      "evaluation_artifact": "evaluation_predictions",
      "parameters_artifact": "dixon_coles_parameters"
    }
  ],
  "release_gate": {
    "status": "passed",
    "baseline_model_id": "bel.1:class-prior:2026-08-10",
    "log_loss": 1.012,
    "brier": 0.608,
    "ece": 0.043,
    "draw_recall": 0.19
  }
}
```

This registry is what Model Lab renders. Static PNGs may remain downloadable, but
structured evaluation predictions and explanation values are the source of truth.

