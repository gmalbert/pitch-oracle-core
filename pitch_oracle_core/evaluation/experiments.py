"""Frozen experiment specifications and forward-test access boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    hypothesis: str
    created_at: datetime
    leagues: tuple[str, ...]
    tracks: tuple[str, ...]
    candidates: tuple[str, ...]
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    mandatory_cohorts: tuple[str, ...]
    tuning_window_end: datetime
    forward_test_start: datetime
    forward_test_end: datetime
    paired_block_unit: str
    family_test: str
    promotion_minimum: Mapping[str, object]
    max_tail_mass: float = 1e-8

    def __post_init__(self) -> None:
        timestamps = (
            self.created_at, self.tuning_window_end,
            self.forward_test_start, self.forward_test_end,
        )
        if any(value.tzinfo is None for value in timestamps):
            raise ValueError("experiment timestamps must be timezone-aware")
        if not self.experiment_id or not self.hypothesis or not self.candidates:
            raise ValueError("experiment ID, hypothesis, and candidates are required")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("experiment candidates must be unique")
        if not self.tuning_window_end < self.forward_test_start < self.forward_test_end:
            raise ValueError("tuning and untouched forward windows overlap")
        if self.max_tail_mass < 0:
            raise ValueError("max_tail_mass cannot be negative")

    @property
    def spec_hash(self) -> str:
        payload = asdict(self)
        for key in ("created_at", "tuning_window_end", "forward_test_start", "forward_test_end"):
            payload[key] = payload[key].isoformat()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=list)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def experiment_partition(
    rows: pd.DataFrame,
    spec: ExperimentSpec,
    *,
    purpose: str,
    time_column: str = "kickoff_utc",
) -> pd.DataFrame:
    """Return tuning or forward rows without allowing cross-boundary access."""
    if purpose not in {"tuning", "forward_test"}:
        raise ValueError("purpose must be tuning or forward_test")
    frame = rows.copy()
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
    if purpose == "tuning":
        return frame.loc[frame[time_column] <= pd.Timestamp(spec.tuning_window_end)].copy()
    return frame.loc[
        (frame[time_column] >= pd.Timestamp(spec.forward_test_start))
        & (frame[time_column] <= pd.Timestamp(spec.forward_test_end))
    ].copy()


def write_experiment_log(
    destination: str | Path,
    spec: ExperimentSpec,
    candidate_results: Mapping[str, Mapping[str, object]],
) -> Path:
    missing = set(spec.candidates).difference(candidate_results)
    unexpected = set(candidate_results).difference(spec.candidates)
    if missing or unexpected:
        raise ValueError(
            f"experiment result coverage differs; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    payload = {
        "experiment_id": spec.experiment_id,
        "spec_hash": spec.spec_hash,
        "candidates": [
            {"model_id": candidate, **dict(candidate_results[candidate])}
            for candidate in spec.candidates
        ],
        "thresholds": dict(spec.promotion_minimum),
        "primary_metric": spec.primary_metric,
        "secondary_metrics": list(spec.secondary_metrics),
        "family_test": spec.family_test,
        "forward_test_range": [
            spec.forward_test_start.isoformat(), spec.forward_test_end.isoformat()
        ],
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
