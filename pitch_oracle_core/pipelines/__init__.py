"""Explicit, import-safe consumer artifact orchestration."""

from .build_consumer import StageReport, atomic_output, build_consumer

__all__ = ["StageReport", "atomic_output", "build_consumer"]
