"""Build and validate the runtime artifact manifest in CI."""

import os

from pitch_oracle_core.cache import validate_cache, write_cache_manifest


if __name__ == "__main__":
    manifest = write_cache_manifest(league=os.getenv("PITCH_ORACLE_LEAGUE"))
    validate_cache()
    print(f"Cache manifest written: {manifest}")
