"""Out-of-fold probability stacker."""

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression

from pitch_oracle_core.evaluation.scores import multiclass_log_loss


@dataclass
class ProbabilityStacker:
    estimator: LogisticRegression | None = None
    component_names: tuple[str, ...] = ()

    @staticmethod
    def _matrix(predictions: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
        names = tuple(sorted(predictions))
        if not names:
            raise ValueError("At least one component is required")
        arrays: list[np.ndarray] = []
        row_count: int | None = None
        for name in names:
            probability = np.asarray(predictions[name], dtype=float)
            if probability.ndim != 2 or probability.shape[1] != 3:
                raise ValueError(f"{name} must have shape (rows, 3)")
            probability = np.clip(probability, 1e-8, 1.0)
            probability /= probability.sum(axis=1, keepdims=True)
            row_count = len(probability) if row_count is None else row_count
            if len(probability) != row_count:
                raise ValueError("Component row counts disagree")
            arrays.append(np.log(probability))
        return np.hstack(arrays), names

    def fit(self, oof_predictions: dict[str, np.ndarray], y: np.ndarray):
        matrix, names = self._matrix(oof_predictions)
        self.estimator = LogisticRegression(
            penalty="l2", C=0.2, solver="lbfgs", max_iter=2_000
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


@dataclass
class CalibratedEnsemble:
    """Use an OOF stack when supported, otherwise the best gated component."""

    minimum_stacking_rows: int = 120
    stacker: ProbabilityStacker | None = None
    fallback_component: str | None = None
    mode: str = "unfitted"

    def fit(self, oof_predictions: dict[str, np.ndarray], y: np.ndarray):
        _, names = ProbabilityStacker._matrix(oof_predictions)
        targets = np.asarray(y, dtype=int)
        if len(targets) == 0 or any(len(oof_predictions[name]) != len(targets) for name in names):
            raise ValueError("OOF predictions and targets disagree")
        self.fallback_component = min(
            names,
            key=lambda name: multiclass_log_loss(targets, oof_predictions[name]),
        )
        if len(targets) < self.minimum_stacking_rows or len(names) < 2:
            self.mode = "best_component_fallback"
            self.stacker = None
            return self
        self.stacker = ProbabilityStacker().fit(oof_predictions, targets)
        self.mode = "oof_calibrated_stack"
        return self

    def predict_proba(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        if self.mode == "unfitted" or self.fallback_component is None:
            raise RuntimeError("CalibratedEnsemble is not fitted")
        if self.mode == "oof_calibrated_stack":
            return self.stacker.predict_proba(predictions)
        if self.fallback_component not in predictions:
            raise ValueError("best fallback component is unavailable")
        probability = np.asarray(predictions[self.fallback_component], dtype=float)
        probability = np.clip(probability, 1e-12, 1.0)
        return probability / probability.sum(axis=1, keepdims=True)
