"""Expected Threat (xT) model — per (competition, season) surface fitting.

Wraps ``penaltyblog.xt.XTModel`` for fitting custom xT surfaces from event
data and scoring events with xT values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from penaltyblog.xt import XTEventSchema, XTModel, load_pretrained_xt


@dataclass(frozen=True)
class XTTrainingResult:
    """Result of fitting an xT model."""
    model_path: Path
    n_events: int
    n_cols: int
    n_rows: int


def fit_xt_model(
    events: pd.DataFrame,
    *,
    schema: XTEventSchema | None = None,
    n_cols: int = 16,
    n_rows: int = 12,
    output_path: str | Path | None = None,
) -> XTModel:
    """Fit an xT model from event data.

    ``events`` must contain columns for x, y, end_x, end_y, event_type,
    and is_success — or a custom ``schema`` must be provided.
    """
    if schema is None:
        schema = XTEventSchema(
            x="x", y="y",
            end_x="end_x", end_y="end_y",
            event_type="event_type",
            is_success="is_success",
            x_range=(0, 100), y_range=(0, 100),
        )
    model = XTModel(n_cols=n_cols, n_rows=n_rows)
    model.fit(events, schema=schema)
    if output_path:
        model.save(str(output_path))
    return model


def score_events(
    model: XTModel,
    events: pd.DataFrame,
    schema: XTEventSchema | None = None,
) -> pd.DataFrame:
    """Score events with xT values, adding xt_start, xt_end, xt_added columns."""
    if schema is None:
        schema = XTEventSchema(
            x="x", y="y",
            end_x="end_x", end_y="end_y",
            event_type="event_type",
            is_success="is_success",
            x_range=(0, 100), y_range=(0, 100),
        )
    return model.score(events, schema=schema)


def pretrained_xt_value(x: float, y: float) -> float:
    """Get xT value from the pretrained model at a given location."""
    model = load_pretrained_xt()
    return model.value_at(x, y)


def xt_features_per_team(
    scored_events: pd.DataFrame,
    team_col: str = "team",
    minutes_col: str = "minutes",
) -> pd.DataFrame:
    """Compute per-team xT features from scored events."""
    agg = scored_events.groupby(team_col).agg(
        total_xt_added=("xt_added", "sum"),
        mean_xt_added=("xt_added", "mean"),
        n_actions=("xt_added", "count"),
    ).reset_index()
    if minutes_col in scored_events.columns:
        minutes = scored_events.groupby(team_col)[minutes_col].sum()
        agg = agg.merge(minutes.reset_index().rename(columns={minutes_col: "total_minutes"}), on=team_col)
        agg["xt_added_per90"] = agg["total_xt_added"] / (agg["total_minutes"] / 90).clip(lower=1)
    return agg
