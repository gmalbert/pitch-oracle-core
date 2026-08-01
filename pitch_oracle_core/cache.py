"""Cache artifact contract shared by every league consumer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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
    CacheRequirement("upcoming_fixtures", "data_files/upcoming_fixtures.csv"),
    CacheRequirement("upcoming_predictions", "data_files/upcoming_predictions.csv"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_cache_manifest(
    root: str | Path = ".",
    requirements: tuple[CacheRequirement, ...] = DEFAULT_REQUIREMENTS,
    league: str | None = None,
) -> Path:
    """Write a manifest proving that CI produced all runtime artifacts."""
    root = Path(root)
    missing = [item.path for item in requirements if not (root / item.path).is_file()]
    if missing:
        raise FileNotFoundError("Missing cache artifacts: " + ", ".join(missing))

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
                "schema_version": 1,
                "league": league,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "artifacts": artifacts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def validate_cache(root: str | Path = ".") -> None:
    """Validate presence and integrity of the CI-produced runtime cache."""
    root = Path(root)
    manifest_path = root / "precomputed" / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cache manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, item in manifest.get("artifacts", {}).items():
        artifact = root / item["path"]
        if not artifact.is_file():
            raise FileNotFoundError(f"Cache artifact '{name}' is missing: {artifact}")
        if artifact.stat().st_size != item["bytes"] or _sha256(artifact) != item["sha256"]:
            raise RuntimeError(f"Cache artifact '{name}' failed integrity validation")
