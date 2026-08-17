"""Run the complete consumer artifact graph as explicit stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import argparse
import importlib
import os
import tempfile


@dataclass(frozen=True)
class StageReport:
    name: str
    rows: int
    output: Path
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.rows < 0:
            raise ValueError("stage row count cannot be negative")


def atomic_output(destination: Path, writer: Callable[[Path], None]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent, suffix=destination.suffix
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        if not temporary.is_file():
            raise RuntimeError("atomic writer did not create its output")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


STAGE_METHODS = (
    "ingest_entities",
    "ingest_fixtures_and_results",
    "validate_canonical_data",
    "build_team_events",
    "build_feature_snapshots",
    "train_and_select_models",
    "build_forecasts_and_explanations",
    "run_season_simulations",
    "build_quality_and_drift_reports",
    "write_and_validate_manifest",
)


def build_consumer(context) -> tuple[StageReport, ...]:
    reports: list[StageReport] = []
    for method_name in STAGE_METHODS:
        method = getattr(context, method_name, None)
        if not callable(method):
            raise TypeError(f"Build context does not implement {method_name}()")
        report = method()
        if not isinstance(report, StageReport):
            raise TypeError(f"{method_name}() did not return StageReport")
        reports.append(report)
    return tuple(reports)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest-version", type=int, default=3, choices=[3])
    parser.add_argument(
        "--context-factory",
        default="consumer_pipeline:create_context",
        help="Import path module:function returning the consumer build context",
    )
    args = parser.parse_args(argv)
    module_name, function_name = args.context_factory.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    context = factory(league=args.league, root=Path(args.root))
    build_consumer(context)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
