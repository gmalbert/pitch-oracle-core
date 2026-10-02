import json
import sys
from types import SimpleNamespace

import pandas as pd

from pitch_oracle_core.pitchapi.pipeline import daily, main
from pitch_oracle_core.pitchapi.storage import read_frame


def test_daily_without_provider_keeps_baseline_and_publishes_honest_health(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PITCH_API_KEY", raising=False)
    directory = tmp_path / "data_files"
    directory.mkdir()
    pd.DataFrame([{"Date": "2025-08-01", "Time": "19:00", "HomeTeam": "Home", "AwayTeam": "Away", "FTHG": 1, "FTAG": 0, "FTR": "H"}, {"Date": "2025-08-08", "Time": "19:00", "HomeTeam": "Away", "AwayTeam": "Home", "FTHG": 0, "FTAG": 0, "FTR": "D"}]).to_csv(directory / "combined_historical_data.csv", index=False)
    # Standalone apps have a different module with this name. Package preparation
    # must retain its own semantics rather than accidentally import that module.
    monkeypatch.setitem(sys.modules, "prepare_model_data", SimpleNamespace())
    daily(league_key="epl", historical_file="pitchapi_historical_features.csv")
    frame = read_frame(directory / "pitchapi_historical_features.csv")
    assert len(frame) == 2
    assert frame.fixture_id.nunique() == 2
    assert frame.home_xg_for_ewm10.isna().all()
    assert json.loads((directory / "pitchapi_health.json").read_text())["mapping"]["gate_passed"] is False
    assert json.loads((directory / "pitchapi_artifacts.json").read_text())["pitchapi"]["enabled_families"] == []
    assert not (directory / "combined_historical_data_with_calculations_new.csv").exists()
    assert main(["index", "--league", "epl"]) == 0
