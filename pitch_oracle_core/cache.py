"""Cache artifact contract shared by every league consumer."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ._version import __version__
from .features import FEATURE_POLICY_VERSION
from .predictions import FeatureContract


@dataclass(frozen=True)
class CacheRequirement:
    name: str
    path: str


DEFAULT_REQUIREMENTS = (
    CacheRequirement("preprocessed_data", "precomputed/preprocessed_data.pkl"),
    CacheRequirement("xgb_baseline", "models/xgb_baseline.pkl"),
    CacheRequirement("ensemble", "models/ensemble_model.pkl"),
    CacheRequirement("optimized_xgb", "models/optimized_xgb.pkl"),
    CacheRequirement("performance", "models/model_performance.pkl"),
    CacheRequirement("model_metadata", "models/model_metadata.json"),
    CacheRequirement("model_audit", "precomputed/model-audit/model_ablation.json"),
    CacheRequirement("upcoming_fixtures", "data_files/upcoming_fixtures.csv"),
    CacheRequirement("upcoming_predictions", "data_files/upcoming_predictions.csv"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_prediction_artifacts(root: Path, requirement_names: set[str]) -> None:
    """Validate semantic compatibility in addition to file integrity."""
    required_group = {"preprocessed_data", "ensemble", "upcoming_predictions"}
    if not required_group.issubset(requirement_names):
        return

    contract = FeatureContract.load(root / "precomputed" / "preprocessed_data.pkl")
    with (root / "models" / "ensemble_model.pkl").open("rb") as stream:
        model = pickle.load(stream)
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and expected_features != len(contract.feature_names):
        raise RuntimeError(
            f"Ensemble expects {expected_features} features but the artifact contract has "
            f"{len(contract.feature_names)}"
        )

    if "model_metadata" in requirement_names:
        metadata = json.loads((root / "models" / "model_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("feature_set") != "no_odds":
            raise RuntimeError("Production model metadata must declare feature_set 'no_odds'")
        if metadata.get("feature_policy_version") != FEATURE_POLICY_VERSION:
            raise RuntimeError("Production model metadata uses a stale feature policy")
        if tuple(metadata.get("feature_names", ())) != contract.feature_names:
            raise RuntimeError("Model metadata and precomputed feature contract disagree")

    if "model_audit" in requirement_names:
        audit = json.loads(
            (root / "precomputed" / "model-audit" / "model_ablation.json").read_text(
                encoding="utf-8"
            )
        )
        if audit.get("status") != "complete" or not audit.get("release_gate", {}).get("passed"):
            raise RuntimeError("Model audit is incomplete or failed its release gate")

    import pandas as pd

    predictions = pd.read_csv(root / "data_files" / "upcoming_predictions.csv", nrows=5)
    required_columns = {
        "HomeWin_Prob", "Draw_Prob", "AwayWin_Prob", "PredictedResult",
        "Risk_Score", "Confidence_Score", "Risk_Category", "Recommendation",
        "ModelLean", "ModelLeanProbability", "BetRecommendation", "BetReason",
        "PredictionGeneratedAt",
    }
    missing = required_columns.difference(predictions.columns)
    if missing:
        raise RuntimeError("Prediction cache is missing columns: " + ", ".join(sorted(missing)))


def write_cache_manifest(
    root: str | Path = ".",
    requirements: tuple[CacheRequirement, ...] = DEFAULT_REQUIREMENTS,
    league: str | None = None,
) -> Path:
    """Write a manifest proving that CI produced all runtime artifacts."""
    root = Path(root)
    active_league = league or os.getenv("PITCH_ORACLE_LEAGUE")
    if not active_league:
        raise ValueError("A league key is required to build a cache manifest")
    missing = [item.path for item in requirements if not (root / item.path).is_file()]
    if missing:
        raise FileNotFoundError("Missing cache artifacts: " + ", ".join(missing))
    _validate_prediction_artifacts(root, {item.name for item in requirements})

    artifacts = {}
    for item in requirements:
        artifact = root / item.path
        artifacts[item.name] = {
            "path": item.path,
            "bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
        }

    output = root / "precomputed" / "cache_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "core_version": __version__,
                "feature_policy_version": FEATURE_POLICY_VERSION,
                "league": active_league,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "artifacts": artifacts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def validate_cache(
    root: str | Path = ".", *, strict_contract: bool = True,
    expected_league: str | None = None,
) -> tuple[str, ...]:
    """Validate a CI-produced runtime cache.

    Integrity failures always raise. ``strict_contract=False`` permits an older
    schema to run temporarily while returning warnings for the UI; CI and cache
    generation should keep the strict default.
    """
    root = Path(root)
    manifest_path = root / "precomputed" / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cache manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    warnings: list[str] = []
    expected_league = expected_league or os.getenv("PITCH_ORACLE_LEAGUE")
    manifest_league = manifest.get("league")
    if not manifest_league:
        message = "Cache manifest is not bound to a league"
        if strict_contract:
            raise RuntimeError(message + "; regenerate runtime artifacts")
        warnings.append(message + " (compatibility mode)")
    elif expected_league and manifest_league != expected_league:
        raise RuntimeError(
            f"Cache belongs to league {manifest_league!r}, not {expected_league!r}"
        )
    contract_current = manifest.get("schema_version") == 2
    if not contract_current:
        message = "Cache manifest schema is stale; predictions are running in compatibility mode"
        if strict_contract:
            raise RuntimeError("Cache manifest schema is stale; regenerate runtime artifacts")
        warnings.append(message)
    elif manifest.get("core_version") != __version__:
        message = (
            f"Cache was built by core {manifest.get('core_version')!r}; {__version__} is running"
        )
        if strict_contract:
            raise RuntimeError(message)
        warnings.append(message + " (compatibility mode)")
        contract_current = False
    elif manifest.get("feature_policy_version") != FEATURE_POLICY_VERSION:
        message = "Cache feature policy is stale; retrain and regenerate runtime artifacts"
        if strict_contract:
            raise RuntimeError(message)
        warnings.append(message + " (compatibility mode)")
        contract_current = False
    for name, item in manifest.get("artifacts", {}).items():
        artifact = root / item["path"]
        if not artifact.is_file():
            raise FileNotFoundError(f"Cache artifact '{name}' is missing: {artifact}")
        if artifact.stat().st_size != item["bytes"] or _sha256(artifact) != item["sha256"]:
            raise RuntimeError(f"Cache artifact '{name}' failed integrity validation")
    if contract_current:
        _validate_prediction_artifacts(root, set(manifest.get("artifacts", {})))
    return tuple(warnings)
