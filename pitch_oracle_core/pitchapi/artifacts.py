"""Optional analytics index shared by legacy and manifest-based consumers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from pitch_oracle_core.artifacts.manifest import file_digest
from .cache import atomic_json
from .contracts import INTEGRATION_SCHEMA_VERSION

FILES = {
    "pitchapi_response_revisions": "pitchapi_response_revisions.parquet",
    "pitchapi_upcoming_predictions": "pitchapi_upcoming_predictions.csv",
    "pitchapi_matches": "pitchapi_matches.csv",
    "pitchapi_shots": "pitchapi_shots.parquet",
    "pitchapi_match_shot_features": "pitchapi_match_shot_features.csv",
    "pitchapi_advanced_team": "pitchapi_advanced_team.parquet",
    "pitchapi_player_match": "pitchapi_player_match.parquet",
    "pitchapi_lineup_snapshots": "pitchapi_lineup_snapshots.parquet",
    "pitchapi_momentum": "pitchapi_momentum.parquet",
    "pitchapi_network": "pitchapi_network.parquet",
    "pitchapi_heatmaps": "pitchapi_heatmaps.parquet",
    "pitchapi_forecast_revisions": "pitchapi_forecast_revisions.parquet",
    "pitchapi_closing_forecasts": "pitchapi_closing_forecasts.parquet",
    "pitchapi_health": "pitchapi_health.json",
    "pitchapi_fixture_audit": "pitchapi_fixture_audit.csv",
    "pitchapi_ablation": "pitchapi_ablation.json",
}


def optional_descriptors(root: str | Path, data_dir: str = "data_files") -> dict:
    root = Path(root).resolve()
    directory = (root / data_dir).resolve()
    directory.relative_to(root)
    descriptors = {}
    for name, filename in FILES.items():
        path = directory / filename
        if not path.is_file():
            continue
        digest, size = file_digest(path)
        descriptors[name] = {"path": path.relative_to(root).as_posix(), "sha256": digest, "bytes": size, "required": False, "schema_version": INTEGRATION_SCHEMA_VERSION}
    return descriptors


def feature_metadata(root: str | Path, league_key: str, data_dir: str = "data_files") -> dict:
    from pitch_oracle_core.features import FEATURE_POLICY_VERSION
    from pitch_oracle_core.features.families import FeatureFamilyConfig
    config = FeatureFamilyConfig.load(Path(root) / data_dir / "pitchapi_feature_config.json", league_key=league_key)
    return {"integration_schema_version": INTEGRATION_SCHEMA_VERSION, "feature_policy_version": FEATURE_POLICY_VERSION, "league_key": league_key, "enabled_families": list(config.enabled_families), "evidence_id": config.evidence_id}


def publish_index(root: str | Path, league_key: str, data_dir: str = "data_files") -> Path:
    root = Path(root)
    path = root / data_dir / "pitchapi_artifacts.json"
    atomic_json(path, {"schema_version": 2, "league": league_key, "generated_at": datetime.now(timezone.utc).isoformat(), "artifacts": optional_descriptors(root, data_dir), "pitchapi": feature_metadata(root, league_key, data_dir)})
    return path


def analytics_repository(root: str | Path, league_key: str, data_dir: str = "data_files"):
    from pitch_oracle_core.ui.repository import ArtifactRepository
    root = Path(root).resolve()
    index = root / data_dir / "pitchapi_artifacts.json"
    if not index.is_file():
        return ArtifactRepository(root, {}, {"league": league_key})
    return ArtifactRepository.from_manifest(root, index.relative_to(root).as_posix(), expected_league=league_key)


def attach_analytics(repository, league_key: str, data_dir: str = "data_files"):
    """Use the separately validated optional index after an hourly refresh."""
    from pitch_oracle_core.ui.repository import ArtifactRepository
    optional = analytics_repository(repository.root, league_key, data_dir)
    failures = {**repository.manifest.get("optional_artifact_failures", {}), **optional.manifest.get("optional_artifact_failures", {})}
    descriptors = dict(repository.descriptors)
    for name in FILES:
        if name in optional.manifest.get("artifacts", {}):
            descriptors.pop(name, None)
    descriptors.update(optional.descriptors)
    return ArtifactRepository(repository.root, descriptors, {**repository.manifest, "optional_artifact_failures": failures, "pitchapi": optional.manifest.get("pitchapi", repository.manifest.get("pitchapi", {}))})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="data_files")
    args = parser.parse_args(argv)
    publish_index(args.root, args.league, args.data_dir)


if __name__ == "__main__":
    main()
