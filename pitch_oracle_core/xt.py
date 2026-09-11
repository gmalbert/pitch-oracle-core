"""Expected Threat (xT) model — per (competition, season) surface fitting.

Wraps ``penaltyblog.xt.XTModel`` for fitting custom xT surfaces from event
data and scoring events with xT values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
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


@dataclass(frozen=True)
class XTArtifactMetadata:
    """Stable metadata for a competition-season xT model artifact."""

    competition_id: str
    season_id: str
    n_events: int
    n_cols: int
    n_rows: int
    source_hash: str
    generated_at: str
    penaltyblog_version: str
    model_metadata: dict

    def as_dict(self) -> dict:
        return {
            "artifact_type": "expected_threat",
            "artifact_version": 1,
            "competition_id": self.competition_id,
            "season_id": self.season_id,
            "n_events": self.n_events,
            "grid": {"n_cols": self.n_cols, "n_rows": self.n_rows},
            "source_hash": self.source_hash,
            "generated_at": self.generated_at,
            "penaltyblog_version": self.penaltyblog_version,
            "model_metadata": self.model_metadata,
        }


def fit_xt_model(
    events: pd.DataFrame,
    *,
    schema: XTEventSchema | None = None,
    n_cols: int = 16,
    n_rows: int = 12,
    output_path: str | Path | None = None,
    competition_id: str | int | None = None,
    season_id: str | int | None = None,
    source_hash: str = "",
) -> XTModel:
    """Fit an xT model from event data.

    ``events`` must contain columns for x, y, end_x, end_y, event_type,
    and is_success — or a custom ``schema`` must be provided.
    """
    if events.empty:
        raise ValueError("cannot fit xT on an empty event frame")
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
    if competition_id is not None or season_id is not None or source_hash:
        model.metadata_["pitch_oracle"] = {
            "competition_id": None if competition_id is None else str(competition_id),
            "season_id": None if season_id is None else str(season_id),
            "source_hash": source_hash,
            "n_events": int(len(events)),
        }
    if output_path:
        model.save(str(output_path))
    return model


def xt_artifact_metadata(
    model: XTModel,
    *,
    competition_id: str | int,
    season_id: str | int,
    n_events: int,
    source_hash: str,
) -> XTArtifactMetadata:
    """Create a serializable descriptor alongside a saved xT model."""
    import penaltyblog

    return XTArtifactMetadata(
        competition_id=str(competition_id),
        season_id=str(season_id),
        n_events=int(n_events),
        n_cols=int(model.n_cols),
        n_rows=int(model.n_rows),
        source_hash=str(source_hash),
        generated_at=datetime.now(timezone.utc).isoformat(),
        penaltyblog_version=str(getattr(penaltyblog, "__version__", "unknown")),
        model_metadata=dict(getattr(model, "metadata_", {})),
    )


def write_xt_artifact_metadata(
    metadata: XTArtifactMetadata,
    destination: str | Path,
) -> Path:
    """Write xT metadata atomically enough for consumer artifact publishing."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


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
