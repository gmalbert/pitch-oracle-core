"""Immutable raw revisions; mutable pointers contain only operational metadata."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable

from .contracts import INTEGRATION_SCHEMA_VERSION, RawObservation, utc_timestamp


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ObservationCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _directory(self, key: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError("Cache key must be an opaque, path-safe identifier")
        return self.root / key

    def latest(self, key: str) -> RawObservation | None:
        directory = self._directory(key)
        index = directory / "latest.json"
        if not index.exists():
            return None
        pointer = json.loads(index.read_text(encoding="utf-8"))
        filename = str(pointer["revision"])
        if Path(filename).name != filename or not filename.endswith(".json"):
            raise ValueError("Invalid cache revision path")
        raw = json.loads((directory / filename).read_text(encoding="utf-8"))
        digest = self._digest(raw["payload"])
        if digest != raw["sha256"] or raw["schema_version"] != INTEGRATION_SCHEMA_VERSION:
            raise ValueError("Raw cache hash or schema mismatch")
        return RawObservation(
            raw["payload"], utc_timestamp(raw["fetched_at"]),
            utc_timestamp(pointer["checked_at"]), raw["endpoint"], digest, True,
        )

    @staticmethod
    def _digest(payload: Any) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def store(self, key: str, endpoint: str, payload: Any, *, now: datetime) -> RawObservation:
        now = utc_timestamp(now)
        previous = self.latest(key)
        digest = self._digest(payload)
        directory = self._directory(key)
        if previous is not None and previous.endpoint != endpoint:
            raise ValueError("Cache key cannot be reused for a different endpoint")
        if previous is not None and now < previous.checked_at:
            raise ValueError("Cache observation clock moved backwards")
        if previous is not None and digest == previous.sha256:
            pointer = json.loads((directory / "latest.json").read_text(encoding="utf-8"))
            pointer["checked_at"] = now.isoformat()
            atomic_json(directory / "latest.json", pointer)
            return replace(previous, checked_at=now, from_cache=False)
        revision = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{digest[:16]}.json"
        raw = {
            "provider": "pitchapi", "schema_version": INTEGRATION_SCHEMA_VERSION,
            "fetched_at": now.isoformat(), "endpoint": endpoint,
            "http_status": 200, "sha256": digest, "payload": payload,
        }
        # The revision is written once and never altered, even after provider corrections.
        destination = directory / revision
        if not destination.exists():
            atomic_json(destination, raw)
        atomic_json(directory / "latest.json", {"revision": revision, "checked_at": now.isoformat()})
        return RawObservation(payload, now, now, endpoint, digest)

    def fetch(
        self, key: str, endpoint: str, loader: Callable[[], Any], *,
        now: datetime | None = None, completed_at: datetime | None = None,
        refresh: bool = False, maximum_age: timedelta = timedelta(days=1),
        correction_days: int = 7, allow_cached_on_error: bool = True,
    ) -> RawObservation:
        now = utc_timestamp(now or datetime.now(timezone.utc))
        previous = self.latest(key)
        if maximum_age.total_seconds() <= 0 or correction_days < 0:
            raise ValueError("Cache age and correction window must be valid")
        if completed_at is not None:
            completed_at = utc_timestamp(completed_at)
        frozen = completed_at is not None and now >= completed_at + timedelta(days=correction_days)
        if previous is not None and not refresh and (frozen or now - previous.checked_at < maximum_age):
            return previous
        try:
            payload = loader()
        except Exception as exc:
            if previous is None or not allow_cached_on_error:
                raise
            return replace(previous, error_code=getattr(exc, "code", type(exc).__name__))
        return self.store(key, endpoint, payload, now=now)

    def revisions(self, key: str, *, as_of: datetime | None = None) -> list[RawObservation]:
        directory = self._directory(key)
        cutoff = utc_timestamp(as_of) if as_of is not None else None
        observations = []
        for path in sorted(directory.glob("*.json")):
            if path.name == "latest.json":
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            observed = utc_timestamp(raw["fetched_at"])
            if cutoff is None or observed <= cutoff:
                digest = self._digest(raw["payload"])
                if digest != raw["sha256"]:
                    raise ValueError("Raw cache revision hash mismatch")
                observations.append(RawObservation(raw["payload"], observed, observed, raw["endpoint"], digest, True))
        return observations
