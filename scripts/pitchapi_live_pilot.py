"""Read-only live schema pilot using previously reconciled fixtures.

Raw responses and normalized samples go to an ignored pilot directory, never
replace a primary fixture schedule, and never authorize model promotion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import dotenv_values

from pitch_oracle_core.pitchapi.cache import ObservationCache, atomic_json
from pitch_oracle_core.pitchapi.client import PitchAPIClient, PitchAPIError
from pitch_oracle_core.pitchapi import normalize as n
from pitch_oracle_core.pitchapi.storage import read_frame, write_frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="precomputed/pitchapi-pilot")
    parser.add_argument("--leagues", nargs="*", default=["belgium", "laliga", "ligue1", "eredivisie", "epl", "scotland", "turkey"])
    args = parser.parse_args(argv)
    key = os.getenv("PITCH_API_KEY") or dotenv_values(".env").get("PITCH_API_KEY")
    client = PitchAPIClient(key, retries=1, timeout=20)
    root = Path(args.root)
    reports = []
    for league in args.leagues:
        directory = root / league
        payload = json.loads((directory / "matches.json").read_text(encoding="utf-8"))
        mapping = read_frame(directory / "provider_fixture_map.csv")
        matches = [m for m in payload["matches"] if m["status"] == "finished" and str(m["id"]) in set(mapping.provider_match_id)]
        if not matches:
            reports.append({"league": league, "status": "no_mapped_completed_fixture"})
            continue
        match = max(matches, key=lambda m: m["time_utc"])
        mapped = mapping.loc[mapping.provider_match_id == match["id"]].iloc[0]
        teams = {str(match["home_team"]["id"]): str(mapped.home_team_id), str(match["away_team"]["id"]): str(mapped.away_team_id)}
        cache = ObservationCache(directory / "raw")
        for endpoint, normalizer in (("shots", n.normalize_shots), ("advanced", n.normalize_team_data), ("advanced/players", n.normalize_player_data), ("advanced/network", n.normalize_network), ("heatmaps", n.normalize_heatmaps), ("lineups", n.normalize_lineups), ("momentum", None), ("players", None)):
            try:
                response = client._request(f"matches/{match['id']}/{endpoint}")
                observed = datetime.now(timezone.utc)
                cache.store(endpoint.replace("/", "_") + "_" + match["id"], endpoint, response, now=observed)
                lineage = {"fixture_id": mapped.fixture_id, "match_id": match["id"], "team_ids": teams, "observed_at": observed}
                if endpoint == "advanced":
                    lineage["home_provider_id"] = str(match["home_team"]["id"])
                if endpoint == "lineups":
                    frame = normalizer(response, fixture_id=mapped.fixture_id, match_id=match["id"], home_team_id=mapped.home_team_id, away_team_id=mapped.away_team_id, snapshot_at=observed, kickoff_utc=pd.Timestamp(match["time_utc"]))
                elif normalizer:
                    frame = normalizer(response, **lineage)
                else:
                    frame = pd.DataFrame(response if isinstance(response, list) else response.get("points" if endpoint == "momentum" else "players", []))
                if not frame.empty:
                    write_frame(frame, directory / f"sample_{endpoint.replace('/', '_')}.parquet")
                report = {"league": league, "endpoint": endpoint, "status": "normalized", "rows": len(frame), "fields": sorted(response) if isinstance(response, dict) else "array", "observed_at": observed.isoformat()}
            except (PitchAPIError, ValueError, KeyError, TypeError, OSError) as exc:
                report = {"league": league, "endpoint": endpoint, "status": "unavailable" if getattr(exc, "code", None) in {"ANALYTICS_UNAVAILABLE", "RESOURCE_NOT_FOUND"} else "failed", "code": getattr(exc, "code", type(exc).__name__)}
            reports.append(report)
            print(f"{league} {endpoint}: {report['status']} {report.get('rows', report.get('code', ''))}", flush=True)
            atomic_json(root / "schema-summary.json", {"reports": reports})
    return 1 if any(row["status"] == "failed" for row in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
