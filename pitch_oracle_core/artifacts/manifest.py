"""Artifact manifest v3 with semantic schemas, hashes, and dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile


@dataclass(frozen=True)
class ArtifactDescriptor:
    name: str
    path: str
    media_type: str
    schema_name: str
    schema_version: int
    rows: int | None
    min_event_time: str | None
    max_event_time: str | None
    generated_at: str
    producer: str
    producer_version: str = ""
    coverage: float | None = None
    freshness_status: str = "unknown"
    freshness_slo_seconds: int | None = None
    fresh_until: str | None = None
    dependencies: tuple[str, ...] = ()
    model_id: str | None = None
    rules_version: str | None = None
    sha256: str = ""
    bytes: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.path or self.schema_version < 1:
            raise ValueError("artifact name, path, and positive schema version are required")
        candidate = Path(self.path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("artifact paths must remain relative to the bundle root")
        if self.rows is not None and self.rows < 0:
            raise ValueError("artifact row count cannot be negative")
        if self.coverage is not None and not 0 <= self.coverage <= 1:
            raise ValueError("artifact coverage must be in [0, 1]")
        if self.freshness_status not in {"fresh", "stale", "unknown", "unavailable"}:
            raise ValueError("invalid artifact freshness status")
        if self.freshness_slo_seconds is not None and self.freshness_slo_seconds <= 0:
            raise ValueError("freshness SLO must be positive")


@dataclass(frozen=True)
class ManifestV3:
    league: str
    edition_id: str
    core_version: str
    entity_registry_version: str
    generated_at: str
    artifacts: tuple[ArtifactDescriptor, ...]
    rules_version: str = ""
    previous_manifest: str | None = None
    capabilities: tuple[dict[str, object], ...] = ()
    feature_statuses: tuple[dict[str, object], ...] = ()
    schema_version: int = 3

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ValueError("ManifestV3 schema_version must be 3")


def file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest(), path.stat().st_size


def validate_dependency_graph(manifest: ManifestV3) -> None:
    names = {item.name for item in manifest.artifacts}
    if len(names) != len(manifest.artifacts):
        raise ValueError("Artifact names must be unique")
    graph = {item.name: set(item.dependencies) for item in manifest.artifacts}
    for name, dependencies in graph.items():
        missing = dependencies.difference(names)
        if missing:
            raise ValueError(f"{name} has missing dependencies: {missing}")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError(f"Artifact dependency cycle includes {name}")
        if name in visited:
            return
        visiting.add(name)
        for dependency in graph[name]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in graph:
        visit(name)


def validate_artifact_files(manifest: ManifestV3, root: str | Path) -> None:
    root = Path(root).resolve()
    for descriptor in manifest.artifacts:
        artifact = (root / descriptor.path).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Artifact escapes bundle root: {descriptor.path}") from exc
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        digest, size = file_digest(artifact)
        if descriptor.sha256 and descriptor.sha256 != digest:
            raise RuntimeError(f"Artifact {descriptor.name} failed hash validation")
        if descriptor.bytes and descriptor.bytes != size:
            raise RuntimeError(f"Artifact {descriptor.name} failed size validation")


def _parse_utc(value: str, field: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def validate_manifest_metadata(manifest: ManifestV3, *, strict: bool = False) -> None:
    """Validate the domain-level provenance promised by manifest schema v3."""
    _parse_utc(manifest.generated_at, "manifest.generated_at")
    if not manifest.league or not manifest.edition_id or not manifest.core_version:
        raise ValueError("manifest league, edition, and core version are required")
    if strict and not manifest.rules_version:
        raise ValueError("strict manifest requires a rules version")
    for item in manifest.artifacts:
        _parse_utc(item.generated_at, f"{item.name}.generated_at")
        if item.min_event_time is not None:
            minimum = _parse_utc(item.min_event_time, f"{item.name}.min_event_time")
        else:
            minimum = None
        if item.max_event_time is not None:
            maximum = _parse_utc(item.max_event_time, f"{item.name}.max_event_time")
        else:
            maximum = None
        if minimum is not None and maximum is not None and maximum < minimum:
            raise ValueError(f"{item.name} event-time coverage is reversed")
        if item.fresh_until is not None:
            _parse_utc(item.fresh_until, f"{item.name}.fresh_until")
        if strict:
            missing = []
            for field, value in (
                ("producer_version", item.producer_version),
                ("rules_version", item.rules_version),
                ("min_event_time", item.min_event_time),
                ("max_event_time", item.max_event_time),
                ("fresh_until", item.fresh_until),
            ):
                if value in (None, ""):
                    missing.append(field)
            if item.rows is None:
                missing.append("rows")
            if item.coverage is None:
                missing.append("coverage")
            if item.freshness_slo_seconds is None:
                missing.append("freshness_slo_seconds")
            if item.freshness_status == "unknown":
                missing.append("freshness_status")
            if len(item.sha256) != 64 or item.bytes <= 0:
                missing.append("hash/bytes")
            if missing:
                raise ValueError(
                    f"Artifact {item.name} lacks strict metadata: {sorted(missing)}"
                )
    if strict:
        feature_ids = [str(item.get("feature_id")) for item in manifest.feature_statuses]
        expected = [f"F{index:02d}" for index in range(1, 51)]
        if sorted(feature_ids) != expected:
            raise ValueError("strict manifest requires exactly F01 through F50 statuses")
        allowed = {"shipped", "intentionally_deferred", "capability_unavailable"}
        invalid = [
            item for item in manifest.feature_statuses
            if item.get("status") not in allowed or not item.get("reason")
        ]
        if invalid:
            raise ValueError("strict feature statuses require an allowed status and reason")


def write_manifest(manifest: ManifestV3, destination: str | Path) -> Path:
    validate_dependency_graph(manifest)
    validate_manifest_metadata(manifest)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent, suffix=".json.tmp"
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def publish_manifest(
    manifest: ManifestV3,
    destination: str | Path,
    *,
    root: str | Path,
) -> Path:
    """Activate a validated immutable graph while retaining one valid predecessor."""
    destination = Path(destination)
    root = Path(root).resolve()
    validate_dependency_graph(manifest)
    validate_manifest_metadata(manifest, strict=True)
    validate_artifact_files(manifest, root)
    f48 = next(
        (item for item in manifest.feature_statuses if item.get("feature_id") == "F48"),
        None,
    )
    if f48 and f48.get("status") == "shipped":
        quality_descriptor = next(
            (item for item in manifest.artifacts if item.name == "quality_report"), None
        )
        if quality_descriptor is None:
            raise RuntimeError("Publication blocked: shipped F48 requires quality_report")
        from pitch_oracle_core.data.validation import require_quality_report_payload

        quality = json.loads((root / quality_descriptor.path).read_text(encoding="utf-8"))
        require_quality_report_payload(quality)
    previous_relative: str | None = None
    if destination.is_file():
        try:
            current = load_manifest(destination)
            validate_manifest_metadata(current, strict=True)
            validate_artifact_files(current, root)
        except (OSError, TypeError, ValueError, RuntimeError):
            current = None
        if current is not None:
            if _parse_utc(manifest.generated_at, "manifest.generated_at") <= _parse_utc(
                current.generated_at, "current.generated_at"
            ):
                raise RuntimeError("Refusing to downgrade or overwrite a current artifact graph")
            previous = destination.with_name("cache_manifest.previous.json")
            shutil.copy2(destination, previous)
            previous_relative = str(previous.resolve().relative_to(root)).replace("\\", "/")
    activated = replace(manifest, previous_manifest=previous_relative)
    return write_manifest(activated, destination)


def load_manifest(path: str | Path) -> ManifestV3:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 3:
        raise ValueError("manifest is not schema v3")
    payload["artifacts"] = tuple(
        ArtifactDescriptor(
            **{
                **item,
                "dependencies": tuple(item.get("dependencies", ())),
            }
        )
        for item in payload.get("artifacts", ())
    )
    payload["capabilities"] = tuple(payload.get("capabilities", ()))
    payload["feature_statuses"] = tuple(payload.get("feature_statuses", ()))
    manifest = ManifestV3(**payload)
    validate_dependency_graph(manifest)
    validate_manifest_metadata(manifest)
    return manifest


def descriptor_for_file(
    *,
    root: str | Path,
    name: str,
    path: str,
    media_type: str,
    schema_name: str,
    schema_version: int,
    rows: int | None,
    generated_at: datetime,
    producer: str,
    producer_version: str | None = None,
    dependencies: tuple[str, ...] = (),
    min_event_time: str | None = None,
    max_event_time: str | None = None,
    coverage: float = 1.0,
    freshness_status: str = "fresh",
    freshness_slo_seconds: int = 604_800,
    fresh_until: str | None = None,
    model_id: str | None = None,
    rules_version: str | None = None,
) -> ArtifactDescriptor:
    artifact = Path(root) / path
    digest, size = file_digest(artifact)
    generated_utc = generated_at.astimezone(timezone.utc)
    if producer_version is None:
        from pitch_oracle_core._version import __version__

        producer_version = __version__
    coverage_start = min_event_time or generated_utc.isoformat()
    coverage_end = max_event_time or generated_utc.isoformat()
    fresh_through = fresh_until or (
        generated_utc + timedelta(seconds=freshness_slo_seconds)
    ).isoformat()
    return ArtifactDescriptor(
        name=name,
        path=path,
        media_type=media_type,
        schema_name=schema_name,
        schema_version=schema_version,
        rows=rows,
        min_event_time=coverage_start,
        max_event_time=coverage_end,
        generated_at=generated_utc.isoformat(),
        producer=producer,
        producer_version=producer_version,
        coverage=coverage,
        freshness_status=freshness_status,
        freshness_slo_seconds=freshness_slo_seconds,
        fresh_until=fresh_through,
        dependencies=dependencies,
        model_id=model_id,
        rules_version=rules_version,
        sha256=digest,
        bytes=size,
    )
