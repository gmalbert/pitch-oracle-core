"""Coverage denominators and optional-provider audit semantics."""

from datetime import datetime, timezone
import pandas as pd

from pitch_oracle_core.pitchapi.coverage import build_coverage_report, capability_from_frame
from pitch_oracle_core.pitchapi.storage import write_frame

NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


def test_coverage_counts_distinct_eligible_ids_not_raw_row_counts():
    report = capability_from_frame(name="shots", expected_ids={"one", "two"}, observed_ids={"one", "old", "extra"}, observed_at=NOW.isoformat(), now=NOW)
    assert report["coverage"] == .5
    assert report["available"] == 1
    assert report["status"] == "degraded"


def test_empty_provider_reports_each_capability_as_unavailable(tmp_path):
    report = build_coverage_report(tmp_path, league_key="eredivisie", now=NOW)
    assert len(report["capabilities"]) == 8
    assert all(item["status"] == "unavailable" for item in report["capabilities"].values())


def test_full_fixture_count_does_not_hide_missing_second_team(tmp_path):
    matches = pd.DataFrame([{"match_id": "m_one", "fixture_id": "fx:one", "kickoff_utc": "2026-09-30T18:00:00Z", "status": "finished", "home_team_id": "tm:one", "away_team_id": "tm:two", "season": "2026/2027", "score_home": 1, "score_away": 0, "observed_at": NOW.isoformat()}])
    write_frame(matches, tmp_path / "pitchapi_matches.csv")
    write_frame(pd.DataFrame([{"fixture_id": "fx:one", "team_id": "tm:one", "observed_at": NOW.isoformat(), "xt_total": 1}]), tmp_path / "pitchapi_advanced_team.parquet")
    report = build_coverage_report(tmp_path, league_key="eredivisie", now=NOW)
    assert report["capabilities"]["advanced_team"]["coverage"] == .5
    assert report["capabilities"]["advanced_team"]["status"] == "degraded"
    assert report["missingness"]["advanced_team"]["season"][0]["count"] == 1
