"""Format-aware, v2/v3-compatible artifact reader without UI dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .manifest import load_manifest, validate_artifact_files


@dataclass(frozen=True)
class ArtifactRepository:
    root: Path
    descriptors: dict[str, dict]
    manifest: dict

    @classmethod
    def from_manifest(
        cls,
        root: str | Path,
        manifest_path: str = "precomputed/cache_manifest.json",
        *,
        expected_league: str | None = None,
        expected_edition: str | None = None,
    ) -> "ArtifactRepository":
        root = Path(root).resolve()
        primary = (root / manifest_path).resolve()
        primary_payload: dict = {}
        primary_error: Exception | None = None
        try:
            primary_payload = json.loads(primary.read_text(encoding="utf-8"))
            if primary_payload.get("schema_version") == 3:
                typed = load_manifest(primary)
                validate_artifact_files(typed, root)
            manifest = primary_payload
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError, RuntimeError) as exc:
            primary_error = exc
            candidates = []
            if isinstance(primary_payload, dict) and primary_payload.get("previous_manifest"):
                candidates.append(str(primary_payload["previous_manifest"]))
            candidates.append("precomputed/cache_manifest.previous.json")
            manifest = None
            anchor_league = expected_league or primary_payload.get("league")
            anchor_edition = expected_edition or primary_payload.get("edition_id")
            for relative in dict.fromkeys(candidates):
                candidate = (root / relative).resolve()
                try:
                    candidate.relative_to(root)
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                    typed = load_manifest(candidate)
                    validate_artifact_files(typed, root)
                except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError, RuntimeError):
                    continue
                if not anchor_league or not anchor_edition:
                    continue
                if (
                    payload.get("league") != anchor_league
                    or payload.get("edition_id") != anchor_edition
                ):
                    continue
                manifest = payload
                manifest["serving_fallback"] = {
                    "manifest": str(candidate.relative_to(root)).replace("\\", "/"),
                    "primary_error": str(primary_error),
                }
                break
            if manifest is None:
                raise RuntimeError(
                    f"No fully valid artifact manifest is available: {primary_error}"
                ) from primary_error
        if expected_league and manifest.get("league") != expected_league:
            raise RuntimeError(
                f"Artifact bundle belongs to {manifest.get('league')!r}, not {expected_league!r}"
            )
        if expected_edition and manifest.get("edition_id") != expected_edition:
            raise RuntimeError(
                "Fallback across competition editions is forbidden: "
                f"{manifest.get('edition_id')!r} != {expected_edition!r}"
            )
        artifacts = manifest.get("artifacts", {})
        if isinstance(artifacts, list):
            descriptors = {item["name"]: item for item in artifacts}
        else:
            descriptors = {name: {"name": name, **item} for name, item in artifacts.items()}
        return cls(root, descriptors, manifest)

    def path(self, name: str) -> Path:
        try:
            relative = self.descriptors[name]["path"]
        except KeyError as exc:
            raise KeyError(f"Artifact {name!r} is unavailable") from exc
        root = self.root.resolve()
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Artifact path escapes repository root: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def frame(self, name: str) -> pd.DataFrame:
        path = self.path(name)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".csv", ".tsv"}:
            return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
        if suffix in {".json", ".jsonl"}:
            if suffix == ".jsonl":
                return pd.read_json(path, lines=True)
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, list):
                return pd.DataFrame(value)
            raise TypeError(f"JSON artifact {name!r} is not tabular")
        raise ValueError(f"Unsupported tabular artifact format: {suffix}")

    def json(self, name: str) -> dict:
        value = json.loads(self.path(name).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"JSON artifact {name!r} is not an object")
        return value

    def arrays(self, name: str) -> dict[str, np.ndarray]:
        path = self.path(name)
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def available(self, name: str) -> bool:
        if name not in self.descriptors:
            return False
        try:
            return self.path(name).is_file()
        except (FileNotFoundError, ValueError):
            return False

    @property
    def capabilities(self) -> dict[str, dict]:
        raw = self.manifest.get("capabilities", {})
        if isinstance(raw, list):
            return {item["name"]: item for item in raw}
        return raw
