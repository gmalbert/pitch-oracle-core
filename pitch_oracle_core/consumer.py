"""Consumer-facing application and pipeline facade."""

from .app_factory import run
from .predictions import FeatureContract, build_prediction_frame, build_upcoming_feature_matrix
from .training import evaluate, train

__all__ = [
    "FeatureContract", "build_prediction_frame", "build_upcoming_feature_matrix",
    "evaluate", "run", "train",
]
