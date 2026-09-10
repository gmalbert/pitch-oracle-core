"""penaltyblog-backed goal-model training with time-decay weights.

The ``DixonColesGoalModel`` fixes the systematic under-pricing of low-score
scorelines (0-0, 1-0, 0-1, 1-1) that the legacy Poisson baseline suffered
from, and runs ~50× faster thanks to its Cython core.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from penaltyblog.models import DixonColesGoalModel, dixon_coles_weights

DEFAULT_XI = 0.0018


def time_decay_weights(
    dates: pd.Series | np.ndarray,
    xi: float = DEFAULT_XI,
    base_date: object | None = None,
) -> np.ndarray:
    """Compute Dixon-Coles time-decay weights for a sequence of match dates.

    More recent matches receive weights closer to 1; older matches decay
    exponentially controlled by ``xi`` (smaller = slower decay).
    """
    return dixon_coles_weights(dates, xi=xi, base_date=base_date)


def training_set_hash(frame: pd.DataFrame) -> str:
    """Return a stable SHA-256 hex digest of a training-set frame."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame).values.tobytes()
    ).hexdigest()


def fit_dixon_coles(
    frame: pd.DataFrame,
    *,
    xi: float = DEFAULT_XI,
    minimizer_options: dict[str, Any] | None = None,
) -> DixonColesGoalModel:
    """Fit a ``DixonColesGoalModel`` with time-decay weights.

    ``frame`` must contain columns ``goals_home``, ``goals_away``,
    ``team_home``, ``team_away``, and ``datetime`` (or ``date``).
    """
    date_col = "datetime" if "datetime" in frame.columns else "date"
    dates = frame[date_col]
    weights = time_decay_weights(dates, xi=xi)
    model = DixonColesGoalModel(
        goals_home=frame["goals_home"].to_numpy().copy(),
        goals_away=frame["goals_away"].to_numpy().copy(),
        teams_home=frame["team_home"].to_numpy().copy(),
        teams_away=frame["team_away"].to_numpy().copy(),
        weights=weights.copy() if hasattr(weights, 'copy') else weights,
    )
    model.fit(minimizer_options=minimizer_options or {"maxiter": 5000, "ftol": 1e-9})
    return model


@dataclass(frozen=True)
class GoalModelTrainingResult:
    model_path: Path
    metadata_path: Path
    training_set_hash: str
    xi: float
    n_fixtures: int
    n_teams: int


def save_goal_model(
    model: DixonColesGoalModel,
    frame: pd.DataFrame,
    models_dir: str | Path,
    xi: float = DEFAULT_XI,
) -> GoalModelTrainingResult:
    """Persist a fitted goal model and its metadata sidecar."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "dixon_coles_goal_model.pkl"
    metadata_path = models_dir / "dixon_coles_goal_model.json"
    model.save(str(model_path))
    hash_value = training_set_hash(
        frame[["goals_home", "goals_away", "team_home", "team_away"]]
    )
    metadata = {
        "model_type": "DixonColesGoalModel",
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "xi": xi,
        "training_set_hash": hash_value,
        "n_fixtures": len(frame),
        "n_teams": len(set(frame["team_home"]).union(frame["team_away"])),
        "params": {k: round(float(v), 6) for k, v in model.get_params().items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return GoalModelTrainingResult(
        model_path=model_path,
        metadata_path=metadata_path,
        training_set_hash=hash_value,
        xi=xi,
        n_fixtures=len(frame),
        n_teams=metadata["n_teams"],
    )


def load_goal_model(path: str | Path) -> DixonColesGoalModel:
    """Load a persisted ``DixonColesGoalModel``."""
    return DixonColesGoalModel.load(str(path))


def goals_frame_from_historical(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the columns needed for goal-model training from the historical CSV.

    The input may be either the football-data source schema or a consumer's
    processed history schema. Both are accepted so the artifact contract does
    not depend on a league-specific column-renaming convention.
    """
    aliases = {
        "HomeTeam": ("HomeTeam",),
        "AwayTeam": ("AwayTeam",),
        "FTHG": ("FTHG", "FullTimeHomeGoals"),
        "FTAG": ("FTAG", "FullTimeAwayGoals"),
        "Date": ("Date", "MatchDate"),
    }
    selected = {
        target: next((column for column in candidates if column in df.columns), None)
        for target, candidates in aliases.items()
    }
    missing = [target for target, column in selected.items() if column is None]
    if missing:
        raise ValueError(f"Historical source misses: {sorted(missing)}")
    frame = df[[selected[target] for target in aliases]].dropna().copy()
    frame.columns = ["team_home", "team_away", "goals_home", "goals_away", "date"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date", kind="stable")
    frame["goals_home"] = frame["goals_home"].astype(int)
    frame["goals_away"] = frame["goals_away"].astype(int)
    return frame.reset_index(drop=True)
