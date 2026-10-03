import pandas as pd
import pytest

from pitch_oracle_core.fixtures.registry import assign_fixture_ids


def fixture(identity, kickoff="2026-08-01T18:00:00Z", home="epl:home", away="epl:away", precision="exact"):
    return pd.DataFrame([{"fixture_id": identity, "kickoff_utc": kickoff, "home_team_id": home, "away_team_id": away, "edition_id": "epl:2026", "kickoff_precision": precision, "source_clock_timezone": "UTC"}])


def test_completion_source_and_reschedule_keep_first_canonical_identity(tmp_path):
    path = tmp_path / "registry.json"
    assign_fixture_ids(fixture("epl:espn:one"), path, league_key="epl")
    changed = assign_fixture_ids(fixture("epl:espn:one", "2026-08-03T18:00:00Z"), path, league_key="epl")
    assert changed.fixture_id.iloc[0] == "epl:espn:one"
    history = assign_fixture_ids(fixture("fx:history", "2026-08-03T19:00:00Z"), path, league_key="epl")
    assert history.fixture_id.iloc[0] == "epl:espn:one"
    assert assign_fixture_ids(fixture("fx:history", "2026-08-03T19:30:00Z"), path, league_key="epl", persist=False).fixture_id.iloc[0] == "epl:espn:one"


def test_registry_does_not_merge_reverse_fixture_or_other_edition(tmp_path):
    path = tmp_path / "registry.json"
    assign_fixture_ids(fixture("one"), path, league_key="epl")
    assert assign_fixture_ids(fixture("two", home="epl:away", away="epl:home"), path, league_key="epl").fixture_id.iloc[0] == "two"
    with pytest.raises(ValueError, match="different league"):
        assign_fixture_ids(fixture("one"), path, league_key="laliga")
    with pytest.raises(ValueError, match="oriented teams"):
        assign_fixture_ids(fixture("one", home="epl:new"), path, league_key="epl")


def test_ambiguous_date_only_source_is_rejected(tmp_path):
    path = tmp_path / "registry.json"
    assign_fixture_ids(fixture("one", "2026-08-01T01:00:00Z"), path, league_key="epl")
    assign_fixture_ids(fixture("two", "2026-08-01T22:00:00Z"), path, league_key="epl")
    with pytest.raises(ValueError, match="Ambiguous"):
        assign_fixture_ids(fixture("unknown", "2026-08-01T12:00:00Z", precision="date_only"), path, league_key="epl")
