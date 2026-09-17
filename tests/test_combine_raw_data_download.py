from combine_raw_data import _download_football_data, _loopback_proxy_overrides


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

    monkeypatch.setattr("combine_raw_data.requests.get", fake_get)

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

    monkeypatch.setattr("combine_raw_data.requests.get", fake_get)
    monkeypatch.setattr("combine_raw_data.time.sleep", lambda _: None)

    frame = _download_football_data(
        "https://www.football-data.co.uk/mmz4281/2627/SC0.csv"
    )

    assert calls[-1] == "https://www.football-data.co.uk/mmz4281/2627/SC0.csv"
    assert len(calls) == 4
    assert len(frame) == 1


def test_download_rejects_html_even_with_success_status(monkeypatch):
    monkeypatch.setattr(
        "combine_raw_data.requests.get",
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
        "combine_raw_data.requests.get",
        lambda url, **kwargs: FakeResponse(
            content=b"\xef\xbb\xbf" + csv,
            headers={"Content-Type": "text/csv"},
        ),
    )

    frame = _download_football_data("https://football-data.co.uk/test.csv")

    assert frame.loc[0, "HomeTeam"] == "Clüb"


def test_download_bypasses_dead_loopback_proxy(monkeypatch):
    calls = []
    csv = b"Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/08/26,A,B,1,0,H\n"
    proxy_names = (
        "http_proxy", "https_proxy", "all_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    )
    for name in proxy_names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:80")

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        return FakeResponse(content=csv, headers={"Content-Type": "text/csv"})

    monkeypatch.setattr("combine_raw_data.requests.get", fake_get)

    frame = _download_football_data("https://football-data.co.uk/test.csv")

    assert len(frame) == 1
    assert calls[0]["proxies"] == {"http": "", "https": "", "all": ""}


def test_download_preserves_external_proxy(monkeypatch):
    proxy_names = (
        "http_proxy", "https_proxy", "all_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    )
    for name in proxy_names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")

    assert _loopback_proxy_overrides() is None
