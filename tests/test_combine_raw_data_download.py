import pandas as pd
import pytest

from combine_raw_data import _direct_get, _download_football_data, combine_raw_data


class FakeResponse:
    def __init__(self, *, status_code=200, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.url = "https://football-data.co.uk/test.csv"

    @property
    def text(self):
        return self.content.decode("utf-8", errors="replace")

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def test_download_prefers_bare_hostname(monkeypatch):
    calls = []
    csv = b"Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/08/26,A,B,1,0,H\n"

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse(content=csv, headers={"Content-Type": "text/csv"})

    monkeypatch.setattr("combine_raw_data._direct_get", fake_get)

    frame = _download_football_data(
        "https://www.football-data.co.uk/mmz4281/2627/SC0.csv"
    )

    assert calls == ["https://football-data.co.uk/mmz4281/2627/SC0.csv"]
    assert frame.loc[0, "HomeTeam"] == "A"


def test_download_falls_back_after_www_503(monkeypatch):
    calls = []
    csv = b"Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/08/26,A,B,1,0,H\n"

    def fake_get(url, **kwargs):
        calls.append(url)
        if "football-data.co.uk/" in url and "www." not in url:
            return FakeResponse(
                status_code=503,
                content=b"<html><title>Maintenance</title></html>",
                headers={"Content-Type": "text/html", "Retry-After": "0"},
            )
        return FakeResponse(content=csv, headers={"Content-Type": "text/csv"})

    monkeypatch.setattr("combine_raw_data._direct_get", fake_get)
    monkeypatch.setattr("combine_raw_data.time.sleep", lambda _: None)

    frame = _download_football_data(
        "https://www.football-data.co.uk/mmz4281/2627/SC0.csv"
    )

    assert calls[-1] == "https://www.football-data.co.uk/mmz4281/2627/SC0.csv"
    assert len(calls) == 4
    assert len(frame) == 1


def test_download_rejects_html_even_with_success_status(monkeypatch):
    monkeypatch.setattr(
        "combine_raw_data._direct_get",
        lambda url, **kwargs: FakeResponse(
            content=b"<!doctype html><html>Unavailable</html>",
            headers={"Content-Type": "text/html"},
        ),
    )
    monkeypatch.setattr("combine_raw_data.time.sleep", lambda _: None)

    try:
        _download_football_data("https://football-data.co.uk/test.csv")
    except RuntimeError as exc:
        assert "HTML response" in str(exc)
        assert "content-type=text/html" in str(exc)
    else:
        raise AssertionError("HTML response should not be parsed as CSV")


def test_download_handles_bom_and_cp1252(monkeypatch):
    csv = "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/08/26,Clüb,A,1,0,H\n".encode(
        "cp1252"
    )
    monkeypatch.setattr(
        "combine_raw_data._direct_get",
        lambda url, **kwargs: FakeResponse(
            content=b"\xef\xbb\xbf" + csv,
            headers={"Content-Type": "text/csv"},
        ),
    )

    frame = _download_football_data("https://football-data.co.uk/test.csv")

    assert frame.loc[0, "HomeTeam"] == "Clüb"


def test_download_uses_session_without_ambient_proxy(monkeypatch):
    calls = []
    csv = b"Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/08/26,A,B,1,0,H\n"

    class FakeSession:
        trust_env = True

        def get(self, url, **kwargs):
            calls.append((self.trust_env, url, kwargs))
            return FakeResponse(content=csv, headers={"Content-Type": "text/csv"})

        def close(self):
            calls.append("closed")

    monkeypatch.setattr("combine_raw_data.requests.Session", FakeSession)

    response = _direct_get("https://football-data.co.uk/test.csv", timeout=1)

    assert response.status_code == 200
    assert calls[0][0] is False
    assert calls[0][1] == "https://football-data.co.uk/test.csv"
    assert calls[1] == "closed"


def test_refresh_retains_cached_history_when_provider_is_unavailable(
    monkeypatch, tmp_path, capsys
):
    cached_path = tmp_path / "combined_historical_data.csv"
    cached = pd.DataFrame(
        [{"MatchDate": "2026-08-01", "HomeTeam": "A", "AwayTeam": "B"}]
    )
    cached.to_csv(cached_path, sep="\t", index=False)
    monkeypatch.setattr(
        "combine_raw_data._download_football_data",
        lambda url: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
    )
    monkeypatch.setattr("combine_raw_data.time.sleep", lambda _: None)

    result = combine_raw_data("epl", seasons=("2526",), output_dir=tmp_path)

    pd.testing.assert_frame_equal(result, cached)
    pd.testing.assert_frame_equal(pd.read_csv(cached_path, sep="\t"), cached)
    assert "retaining 1 cached historical rows" in capsys.readouterr().out


def test_refresh_rejects_invalid_cached_history(monkeypatch, tmp_path):
    cached_path = tmp_path / "combined_historical_data.csv"
    cached_path.write_text("not,a,valid,cache\n", encoding="utf-8")
    monkeypatch.setattr(
        "combine_raw_data._download_football_data",
        lambda url: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
    )
    monkeypatch.setattr("combine_raw_data.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="cached historical data is invalid"):
        combine_raw_data("epl", seasons=("2526",), output_dir=tmp_path)
