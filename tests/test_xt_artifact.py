import json

import pandas as pd


def _events() -> pd.DataFrame:
    return pd.DataFrame({
        "x": [10, 20, 80, 90],
        "y": [10, 20, 50, 50],
        "end_x": [20, 30, 90, 95],
        "end_y": [20, 30, 50, 50],
        "event_type": ["pass", "pass", "shot", "shot"],
        "is_success": [True, True, True, True],
    })


def test_xt_fit_carries_competition_and_source_metadata(tmp_path):
    from pitch_oracle_core.xt import (
        fit_xt_model,
        write_xt_artifact_metadata,
        xt_artifact_metadata,
    )

    model = fit_xt_model(
        _events(), n_cols=4, n_rows=3, competition_id=11, season_id=22,
        source_hash="a" * 64,
    )
    metadata = xt_artifact_metadata(
        model, competition_id=11, season_id=22, n_events=4, source_hash="a" * 64
    )
    path = write_xt_artifact_metadata(metadata, tmp_path / "xt.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["competition_id"] == "11"
    assert payload["season_id"] == "22"
    assert payload["source_hash"] == "a" * 64
    assert payload["grid"] == {"n_cols": 4, "n_rows": 3}
