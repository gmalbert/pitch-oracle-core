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
TEXT_SUFFIXES = {"", ".md", ".py", ".txt", ".ini", ".yml", ".yaml", ".json"}
REPOSITORY_SLUGS = {
    "scotland": "scotland-premiership",
    "eredivisie": "netherlands-soccer",
    "portugal": "portugal-soccer",
    "belgium": "belgium-soccer",
    "turkey": "turkey-soccer",
}


def consumer_ready_leagues() -> tuple[str, ...]:
    """Return non-EPL leagues with both baseline providers configured."""
    return tuple(
        sorted(
            key
            for key, config in BUILTIN_LEAGUES.items()
            if key != "epl" and config.football_data_div and config.espn_slug
        )
    )


def repository_slug_for(league_key: str) -> str:
    """Return the required country-based repository name for a league."""
    try:
        return REPOSITORY_SLUGS[league_key.lower()]
    except KeyError as exc:
        raise ValueError(f"No country repository name configured for {league_key!r}") from exc


def copy_local_environment(
    destination: str | Path, source_env: str | Path | None = None
) -> bool:
    """Copy local secrets into a consumer only when Git explicitly ignores them."""
    destination = Path(destination)
    source = Path(source_env) if source_env is not None else CORE_ROOT / ".env"
    if not source.is_file():
        return False
    ignore_path = destination / ".gitignore"
    ignored = {
        line.strip() for line in ignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if ".env" not in ignored:
        raise RuntimeError("Refusing to copy secrets: consumer .gitignore does not ignore .env")
    shutil.copy2(source, destination / ".env")
    return True


def bootstrap_consumer(
    league_key: str,
    parent: str | Path = "..",
    *,
    allow_existing_empty: bool = False,
) -> Path:
    """Create a country-named consumer directory below ``parent``."""
    key = league_key.lower()
    ready = consumer_ready_leagues()
    if key not in ready:
        choices = ", ".join(ready)
        raise ValueError(
            f"League {league_key!r} is not consumer-ready; choose from: {choices}. "
            "Add and test its baseline provider identifiers in pitch-oracle-core first."
        )

    destination = Path(parent).resolve() / repository_slug_for(key)
    if destination.exists() and not allow_existing_empty:
        raise FileExistsError(
            f"Refusing to overwrite existing path: {destination}. "
            "Choose a new, empty repository path."
        )

    if destination.exists():
        existing_files = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file() and "venv" not in path.parts and ".git" not in path.parts
        }
        allowed = {"README.md"} | {
            (Path("data_files") / path.name).as_posix()
            for path in (destination / "data_files").glob("*")
            if path.is_file()
        }
        if not existing_files.issubset(allowed):
            raise FileExistsError(
                f"Existing path is not an empty consumer repository: {destination}"
            )
    shutil.copytree(
        TEMPLATE_ROOT,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        dirs_exist_ok=True,
    )
    config = BUILTIN_LEAGUES[key]
    for path in destination.rglob("*"):
        if (
            not path.is_file()
            or "venv" in path.parts
            or ".git" in path.parts
            or path.suffix.lower() not in TEXT_SUFFIXES
        ):
            continue
        content = path.read_text(encoding="utf-8")
        content = content.replace("eredivisie", config.key)
        content = content.replace("Eredivisie", config.display_name)
        path.write_text(content, encoding="utf-8")

    copy_local_environment(destination)

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a turnkey Pitch Oracle consumer repository."
    )
    parser.add_argument("league_key", choices=consumer_ready_leagues())
    parser.add_argument(
        "parent",
        nargs="?",
        default="..",
        help="Parent directory for the generated country-soccer repository (default: ..)",
    )
    parser.add_argument(
        "--existing-empty",
        action="store_true",
        help="Populate an existing empty repository while preserving data_files assets.",
    )
    args = parser.parse_args()
    destination = bootstrap_consumer(
        args.league_key, args.parent, allow_existing_empty=args.existing_empty
    )
    print(f"Created {args.league_key} consumer at {destination}")
    print("Required next steps:")
    print(f"  1. cd {destination}")
    print("  2. Create a virtual environment and install requirements-ci.txt")
    print("  3. Run: python scripts/bootstrap_local.py")
    print("  4. Run: python -m pytest -q and browser-test the Predictions page")


if __name__ == "__main__":
    main()
