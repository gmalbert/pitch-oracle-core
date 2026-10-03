"""Small transport adapter with bounded retries and machine-readable failures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import time
from typing import Any, Callable, Protocol

import requests

from .cache import ObservationCache

BASE_URL = "https://api.pitchapi.dev/v1"


class PitchAPIError(RuntimeError):
    def __init__(self, code: str, message: str = "", status_code: int | None = None, request_id: str | None = None):
        # Preserve legacy callers that supply a single 'CODE: message' string.
        if not message and ": " in code:
            code, message = code.split(": ", 1)
        super().__init__(f"{code}: {message}")
        self.code = code
        self.status_code = status_code
        self.request_id = request_id


class PitchAPITransport(Protocol):
    def leagues(self) -> list[dict]: ...
    def league_matches(self, league_id: str, season: str | None = None, *, status: str | None = None) -> list[dict]: ...
    def _request(self, path: str, *, params: dict | None = None) -> Any: ...


@dataclass
class PitchAPIClient:
    api_key: str = field(repr=False)
    base_url: str = BASE_URL
    session: requests.Session = field(default_factory=requests.Session, repr=False)
    retries: int = 3
    timeout: float = 30.0
    maximum_retry_delay: float = 30.0
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _cache_dir: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key or self.retries < 0 or self.timeout <= 0 or self.maximum_retry_delay <= 0:
            raise ValueError("A credential and valid request limits are required")
        self.session.headers.update({"X-API-KEY": self.api_key})

    def _retry_delay(self, response, attempt: int) -> float:
        raw = response.headers.get("Retry-After") if response is not None else None
        if raw is not None:
            try:
                delay = float(raw)
            except (TypeError, ValueError):
                try:
                    delay = (parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    delay = 2 ** attempt
        else:
            delay = 2 ** attempt
        return min(self.maximum_retry_delay, max(0.0, delay))

    def _request(self, path: str, *, params: dict | None = None) -> Any:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        for attempt in range(self.retries + 1):
            response = None
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == self.retries:
                    # Do not expose request URLs/headers which can contain credentials.
                    raise PitchAPIError("TRANSPORT_ERROR", type(exc).__name__) from exc
            else:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                    error = payload["error"]
                    failure = PitchAPIError(str(error.get("code", "HTTP_ERROR")), str(error.get("message", "")), response.status_code, response.headers.get("X-Request-ID"))
                elif response.status_code >= 400:
                    failure = PitchAPIError("HTTP_ERROR", f"HTTP {response.status_code}", response.status_code)
                elif not isinstance(payload, dict) or "data" not in payload or not isinstance(payload["data"], (dict, list)):
                    failure = PitchAPIError("MALFORMED_RESPONSE", "Expected a data envelope", response.status_code)
                else:
                    return payload["data"]
                retryable = response.status_code == 429 or 500 <= response.status_code < 600
                if not retryable or attempt == self.retries:
                    raise failure
            self.sleep(self._retry_delay(response, attempt))
        raise RuntimeError("Unreachable retry state")

    def leagues(self) -> list[dict]:
        return self._request("leagues").get("leagues", [])

    def league_matches(self, league_id: str, season: str | None = None, *, status: str | None = None) -> list[dict]:
        params = {}
        if season is not None:
            params["season"] = season
        if status is not None:
            if status not in {"played", "upcoming", "all"}:
                raise ValueError("Invalid match listing status")
            params["status"] = status
        return self._request(f"leagues/{league_id}/matches", params=params).get("matches", [])

    def date(self, date: str, *, status: str | None = None) -> list[dict]:
        return self._request(f"date/{date}", params={"status": status} if status else None).get("matches", [])

    def match(self, match_id: str) -> dict:
        return self._request(f"matches/{match_id}")

    def _optional(self, path: str, *, codes: tuple[str, ...] = ("ANALYTICS_UNAVAILABLE",), params: dict | None = None) -> dict | None:
        try:
            return self._request(path, params=params)
        except PitchAPIError as exc:
            if exc.code in codes:
                return None
            raise

    def shots(self, match_id: str) -> dict:
        return self._request(f"matches/{match_id}/shots")

    def advanced_team(self, match_id: str) -> dict | None:
        return self._optional(f"matches/{match_id}/advanced")

    def advanced_players(self, match_id: str) -> dict | None:
        return self._optional(f"matches/{match_id}/advanced/players")

    def players(self, match_id: str) -> dict:
        return self._request(f"matches/{match_id}/players")

    def lineups(self, match_id: str) -> dict | None:
        return self._optional(f"matches/{match_id}/lineups", codes=("ANALYTICS_UNAVAILABLE", "RESOURCE_NOT_FOUND"))

    def momentum(self, match_id: str) -> dict:
        return self._request(f"matches/{match_id}/momentum")

    def network(self, match_id: str) -> dict | None:
        return self._optional(f"matches/{match_id}/advanced/network")

    def heatmaps(self, match_id: str, *, frame: str = "acting_ltr") -> dict | None:
        if frame not in {"acting_ltr", "home_ltr"}:
            raise ValueError("Unknown heatmap coordinate frame")
        return self._optional(f"matches/{match_id}/heatmaps", params={"frame": frame})

    # Compatibility names used by existing consumers and XGProvider.
    def match_shots(self, match_id: str) -> list[dict]:
        return self.shots(match_id).get("periods", [])

    match_advanced = advanced_team
    match_lineups = lineups

    def match_momentum(self, match_id: str) -> list[dict]:
        return self.momentum(match_id).get("points", [])

    def match_players(self, match_id: str) -> list[dict]:
        payload = self.players(match_id)
        return payload.get("players", []) if isinstance(payload, dict) else payload

    def cache_dir(self, base: Path) -> Path:
        if self._cache_dir is None:
            self._cache_dir = base / "pitchapi_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir

    def get_cached_or_fetch(self, base: Path, cache_key: str, path: str, *, params: dict | None = None) -> Any:
        return ObservationCache(self.cache_dir(base)).fetch(
            cache_key, path, lambda: self._request(path, params=params),
        ).payload
