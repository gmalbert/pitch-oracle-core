"""Strict manifest-v3 verification command."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from pitch_oracle_core._version import __version__
from .manifest import (
    load_manifest,
    validate_artifact_files,
    validate_manifest_metadata,
)


def _observed_rows(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return len(pd.read_parquet(path))
    if suffix in {".csv", ".tsv"}:
        return len(pd.read_csv(path, sep="\t" if suffix == ".tsv" else ","))
    if suffix == ".jsonl":
        return len(pd.read_json(path, lines=True))
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return len(payload) if isinstance(payload, list) else 1
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "fixture_ids" in archive:
                return len(archive["fixture_ids"])
            first = next(iter(archive.files), None)
            return 0 if first is None else len(archive[first])
    raise ValueError(f"Strict verification cannot count rows in {path.name}")


def verify(root: str | Path = ".", *, strict: bool = False) -> None:
    root = Path(root)
    manifest = load_manifest(root / "precomputed" / "cache_manifest.json")
    validate_artifact_files(manifest, root)
    validate_manifest_metadata(manifest, strict=strict)
    if strict:
        if manifest.core_version != __version__:
            raise ValueError(
                f"Manifest core version {manifest.core_version} != runtime {__version__}"
            )
        for descriptor in manifest.artifacts:
            observed = _observed_rows(root / descriptor.path)
            if observed != descriptor.rows:
                raise ValueError(
                    f"Artifact {descriptor.name} declares {descriptor.rows} rows, found {observed}"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    verify(args.root, strict=args.strict)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
