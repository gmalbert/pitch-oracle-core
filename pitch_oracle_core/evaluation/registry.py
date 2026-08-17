"""Champion/challenger decision records and deterministic promotion gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .calibration import expected_calibration_error
from .scores import multiclass_brier, multiclass_log_loss


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    family: str
    track: str
    status: str
    trained_through: str
    evaluation_artifact: str
    parameters_artifact: str | None = None
    feature_set_version: str = ""
    entity_registry_version: str = ""
    rules_version: str = ""
    model_card: str = ""
    reproduction_command: str = ""


@dataclass(frozen=True)
class ReleaseDecision:
    status: str
    champion_model_id: str
    challenger_model_id: str
    primary_score_improved: bool
    paired_probability_better: float
    calibration_passed: bool
    cohorts_passed: bool
    coherence_passed: bool
    operations_passed: bool
    reason: str

    @property
    def promote(self) -> bool:
        return (
            self.status == "passed"
            and self.primary_score_improved
            and self.paired_probability_better >= 0.90
            and self.calibration_passed
            and self.cohorts_passed
            and self.coherence_passed
            and self.operations_passed
        )


def evaluate_release_decision(
    champion: pd.DataFrame,
    challenger: pd.DataFrame,
    *,
    champion_model_id: str,
    challenger_model_id: str,
    block_length: int = 8,
    bootstrap_repetitions: int = 2_000,
    calibration_ece_limit: float = 0.05,
    maximum_cohort_brier_degradation: float = 0.02,
    coherence_passed: bool,
    operations_passed: bool,
    seed: int = 20260810,
) -> ReleaseDecision:
    """Reproduce the contextual promotion gate from persisted OOF rows."""
    probability_columns = ["p_home", "p_draw", "p_away"]
    target_column = "actual_outcome" if "actual_outcome" in champion else "target"
    required = {"fixture_id", target_column, *probability_columns}
    for name, frame in (("champion", champion), ("challenger", challenger)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} rows miss: {sorted(missing)}")
        if frame.fixture_id.duplicated().any():
            raise ValueError(f"{name} fixture IDs are not unique")
    if set(champion.fixture_id) != set(challenger.fixture_id):
        raise ValueError(
            "candidate coverage differs; publish coverage-matched and fallback scores separately"
        )
    joined = champion.merge(
        challenger, on="fixture_id", suffixes=("_champion", "_challenger"),
        validate="one_to_one",
    )
    y_champion = joined[f"{target_column}_champion"].to_numpy(dtype=int)
    y_challenger = joined[f"{target_column}_challenger"].to_numpy(dtype=int)
    if not np.array_equal(y_champion, y_challenger):
        raise ValueError("champion and challenger targets disagree")
    champion_p = joined[[f"{name}_champion" for name in probability_columns]].to_numpy()
    challenger_p = joined[[f"{name}_challenger" for name in probability_columns]].to_numpy()
    champion_log_rows = -np.log(np.clip(champion_p[np.arange(len(joined)), y_champion], 1e-15, 1))
    challenger_log_rows = -np.log(np.clip(challenger_p[np.arange(len(joined)), y_champion], 1e-15, 1))
    advantage = champion_log_rows - challenger_log_rows
    rng = np.random.default_rng(seed)
    probability_better = 0.0
    for _ in range(bootstrap_repetitions):
        starts = rng.integers(0, len(joined), size=int(np.ceil(len(joined) / block_length)))
        sample = ((starts[:, None] + np.arange(block_length)) % len(joined)).ravel()[:len(joined)]
        probability_better += float(advantage[sample].mean() > 0)
    probability_better /= bootstrap_repetitions
    primary_improved = (
        multiclass_log_loss(y_champion, challenger_p)
        < multiclass_log_loss(y_champion, champion_p)
        and multiclass_brier(y_champion, challenger_p)
        < multiclass_brier(y_champion, champion_p)
    )
    eces = [
        expected_calibration_error(challenger_p[:, index], y_champion == index)
        for index in range(3)
    ]
    calibration_passed = max(eces) <= calibration_ece_limit
    cohorts_passed = True
    if "cohort_champion" in joined:
        for _, group in joined.groupby("cohort_champion", observed=True):
            if len(group) < 20:
                continue
            indices = group.index.to_numpy()
            # Merge preserves a range index for normal persisted evaluation rows.
            champion_brier = multiclass_brier(y_champion[indices], champion_p[indices])
            challenger_brier = multiclass_brier(y_champion[indices], challenger_p[indices])
            if challenger_brier - champion_brier > maximum_cohort_brier_degradation:
                cohorts_passed = False
                break
    passed = all((
        primary_improved,
        probability_better >= 0.90,
        calibration_passed,
        cohorts_passed,
        coherence_passed,
        operations_passed,
    ))
    reason = (
        "All chronological score, paired uncertainty, calibration, cohort, "
        "coherence, and operations gates passed."
        if passed else
        "Challenger remains non-production because one or more frozen release gates failed."
    )
    return ReleaseDecision(
        status="passed" if passed else "failed",
        champion_model_id=champion_model_id,
        challenger_model_id=challenger_model_id,
        primary_score_improved=primary_improved,
        paired_probability_better=float(probability_better),
        calibration_passed=calibration_passed,
        cohorts_passed=cohorts_passed,
        coherence_passed=coherence_passed,
        operations_passed=operations_passed,
        reason=reason,
    )


def write_model_registry(
    destination: str | Path,
    *,
    production_model_id: str,
    models: list[ModelRecord],
    decision: ReleaseDecision,
    feature_statuses: dict[str, str] | None = None,
) -> Path:
    if production_model_id not in {model.model_id for model in models}:
        raise ValueError("production model is absent from registry")
    production = next(model for model in models if model.model_id == production_model_id)
    if not production.model_card or not production.reproduction_command:
        raise ValueError("production registry row requires a model card and reproduction command")
    payload = {
        "schema_version": 1,
        "production_model_id": production_model_id,
        "models": [asdict(model) for model in models],
        "release_gate": {**asdict(decision), "promote": decision.promote},
        "feature_statuses": feature_statuses or {},
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
