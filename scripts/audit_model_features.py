"""Backward-compatible wrapper for the packaged model-audit CLI."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pitch_oracle_core.audit_cli import generate, main  # noqa: E402,F401


if __name__ == "__main__":
    main()
