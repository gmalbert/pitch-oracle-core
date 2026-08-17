"""Registered score-model implementations and forecast uncertainty helpers."""

from importlib import import_module

from .elo import EloModel
from .independent_poisson import independent_poisson_grid
from .protocol import FixtureFeatures, ForecastTrack, ModelSpec, ScoreModel


_LAZY_MODELS = {
    "DixonColesForecaster": (".dixon_coles", "DixonColesForecaster"),
    "RankCovariateGoalsModel": (".rank_covariate", "RankCovariateGoalsModel"),
}


def __getattr__(name: str):
    target = _LAZY_MODELS.get(name)
    if target is None:
        raise AttributeError(name)
    module = import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value

__all__ = [
    "DixonColesForecaster", "EloModel", "FixtureFeatures", "ForecastTrack", "ModelSpec", "ScoreModel",
    "RankCovariateGoalsModel", "independent_poisson_grid",
]
