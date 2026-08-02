"""Create a thin Pitch Oracle league consumer from the maintained template."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pitch_oracle_core import BUILTIN_LEAGUES  # noqa: E402


TEMPLATE_ROOT = CORE_ROOT / "templates" / "consumer"
TEXT_SUFFIXES = {"", ".md", ".py", ".txt", ".ini", ".yml", ".yaml"}


def consumer_ready_leagues() -> tuple[str, ...]:
    """Return non-EPL leagues with both baseline providers configured."""
    return tuple(
        sorted(
            key
            for key, config in BUILTIN_LEAGUES.items()
            if key != "epl" and config.football_data_div and config.espn_slug
        )
    )


def bootstrap_consumer(league_key: str, target: str | Path) -> Path:
    """Copy and specialize the consumer template into a new directory."""
    key = league_key.lower()
    ready = consumer_ready_leagues()
    if key not in ready:
        choices = ", ".join(ready)
        raise ValueError(
            f"League {league_key!r} is not consumer-ready; choose from: {choices}. "
            "Add and test its baseline provider identifiers in pitch-oracle-core first."
        )

    destination = Path(target).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing path: {destination}. "
            "Choose a new, empty repository path."
        )

    shutil.copytree(
        TEMPLATE_ROOT,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    config = BUILTIN_LEAGUES[key]
    for path in destination.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        content = path.read_text(encoding="utf-8")
        content = content.replace("eredivisie", config.key)
        content = content.replace("Eredivisie", config.display_name)
        path.write_text(content, encoding="utf-8")

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a turnkey Pitch Oracle consumer repository."
    )
    parser.add_argument("league_key", choices=consumer_ready_leagues())
    parser.add_argument("target", help="New directory to create")
    args = parser.parse_args()
    destination = bootstrap_consumer(args.league_key, args.target)
    print(f"Created {args.league_key} consumer at {destination}")
    print("Next: follow README.md in the generated repository.")


if __name__ == "__main__":
    main()
