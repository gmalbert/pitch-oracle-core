import pandas as pd
import pytest

from pitch_oracle_core.artifacts.repository import ArtifactRepository, read_tabular_frame


@pytest.mark.parametrize("extension,delimiter", [("csv", ","), ("csv", "\t"), ("tsv", "\t"), ("parquet", None)])
def test_fixture_read_filters_rows_and_projects_columns(tmp_path, extension, delimiter):
    frame = pd.DataFrame({"fixture_id": ["one", "two", "one"], "team_id": ["home", "home", "away"], "value": [1., 2., 3.], "unused": [100, 200, 300]})
    path = tmp_path / f"shots.{extension}"
    if extension == "parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, sep=delimiter, index=False)
    repository = ArtifactRepository(tmp_path, {"shots": {"path": path.name}}, {})
    result = repository.fixture_frame("shots", "one", columns=["team_id", "value"])
    pd.testing.assert_frame_equal(result.reset_index(drop=True), frame.loc[frame.fixture_id == "one", ["team_id", "value"]].reset_index(drop=True))
    assert set(repository.frame("shots", filters={"team_id": ["away"]}).fixture_id) == {"one"}


def test_parquet_predicates_are_pushed_to_reader(tmp_path, monkeypatch):
    path = tmp_path / "large.parquet"
    pd.DataFrame({"fixture_id": ["one", "two"], "value": [1, 2]}).to_parquet(path)
    original = pd.read_parquet
    arguments = {}
    def observed(*args, **kwargs):
        arguments.update(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", observed)
    result = read_tabular_frame(path, columns=["value"], filters={"fixture_id": "one"})
    assert arguments["filters"] == [("fixture_id", "=", "one")]
    assert arguments["columns"] == ["value", "fixture_id"]
    assert result.value.tolist() == [1]


def test_missing_filter_column_reports_schema_error(tmp_path):
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"value": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="filter column"):
        read_tabular_frame(path, filters={"fixture_id": "one"})
