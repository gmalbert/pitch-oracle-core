"""Consumer-facing application and pipeline facade."""

from .app_factory import run
from .training import evaluate, train

__all__ = ["evaluate", "run", "train"]
