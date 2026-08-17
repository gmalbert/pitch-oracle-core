"""Safety boundary for model-sensitivity scenario controls."""

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ScenarioControl:
    feature: str
    label: str
    minimum: float
    maximum: float
    step: float


def apply_scenario(base_features, changes, allowed_controls):
    allowed = {control.feature for control in allowed_controls}
    forbidden = set(changes).difference(allowed)
    if forbidden:
        raise ValueError(f"Scenario cannot mutate: {sorted(forbidden)}")
    scenario = base_features.copy()
    for feature, value in changes.items():
        control = next(item for item in allowed_controls if item.feature == feature)
        if not control.minimum <= value <= control.maximum:
            raise ValueError(f"Scenario value outside bounds for {feature}")
        scenario[feature] = value
    return scenario


@dataclass(frozen=True)
class ScenarioInferenceAdapter:
    """Apply allowlisted changes through the deployed model's own predictor.

    The adapter deliberately accepts a predictor callback instead of implementing a
    second approximation in the UI. Consumers that do not ship that callback must
    leave the controls disabled through the ``scenario_inference`` capability.
    """

    base_features: Mapping[str, object]
    controls: Sequence[ScenarioControl]
    predictor: Callable[[Mapping[str, object]], np.ndarray]
    cached_probability: np.ndarray

    def __post_init__(self) -> None:
        self._validate_probability(self.cached_probability)

    @staticmethod
    def _validate_probability(value: np.ndarray) -> np.ndarray:
        probability = np.asarray(value, dtype=float)
        if (
            probability.shape != (3,)
            or not np.isfinite(probability).all()
            or (probability < 0).any()
            or not np.isclose(probability.sum(), 1.0, atol=1e-8)
        ):
            raise ValueError("scenario predictor must return a coherent 1X2 vector")
        return probability

    def predict(self, changes: Mapping[str, float]) -> np.ndarray:
        scenario = apply_scenario(dict(self.base_features), changes, self.controls)
        if scenario == dict(self.base_features):
            return np.asarray(self.cached_probability, dtype=float).copy()
        return self._validate_probability(self.predictor(scenario)).copy()
