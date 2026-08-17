"""Canonical football entities and provider alias resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
import re
import unicodedata


class ResolutionStatus(StrEnum):
    EXACT_CANONICAL = "exact_canonical"
    PROVIDER_ALIAS = "provider_alias"
    NEW_TEAM_PRIOR = "new_team_prior"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Team:
    team_id: str
    canonical_name: str
    country_code: str
    founded: int | None = None

    def __post_init__(self) -> None:
        if not self.team_id or ":" not in self.team_id:
            raise ValueError("team_id must be a non-empty namespaced identifier")
        if not self.canonical_name.strip():
            raise ValueError("canonical_name is required")


@dataclass(frozen=True)
class TeamAlias:
    provider: str
    external_name: str
    team_id: str
    valid_from: date | None = None
    valid_to: date | None = None

    def __post_init__(self) -> None:
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("alias validity interval is reversed")

    def valid_on(self, when: date) -> bool:
        return (
            (self.valid_from is None or self.valid_from <= when)
            and (self.valid_to is None or when <= self.valid_to)
        )


@dataclass(frozen=True)
class Resolution:
    raw_name: str
    team_id: str | None
    status: ResolutionStatus
    provider: str


def normalized_name(value: str) -> str:
    """Normalize for lookup while preserving the source display value elsewhere."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


class EntityResolver:
    """Resolve only reviewed canonical names and exact provider aliases.

    Fuzzy matching is deliberately excluded: production coverage failures must be
    reviewed instead of silently joining two clubs with similar names.
    """

    def __init__(self, teams: list[Team], aliases: list[TeamAlias]) -> None:
        self._teams = {team.team_id: team for team in teams}
        if len(self._teams) != len(teams):
            raise ValueError("team_id values must be unique")
        self._canonical: dict[str, str] = {}
        for team in teams:
            key = normalized_name(team.canonical_name)
            if key in self._canonical:
                raise ValueError(f"Ambiguous canonical team name {team.canonical_name!r}")
            self._canonical[key] = team.team_id
        self._aliases: dict[tuple[str, str], list[TeamAlias]] = {}
        for alias in aliases:
            if alias.team_id not in self._teams:
                raise ValueError(f"Alias references unknown team {alias.team_id!r}")
            key = (alias.provider.casefold(), normalized_name(alias.external_name))
            self._aliases.setdefault(key, []).append(alias)

    @property
    def teams(self) -> tuple[Team, ...]:
        return tuple(self._teams.values())

    def resolve(self, provider: str, raw_name: str, when: date) -> Resolution:
        candidates = [
            alias
            for alias in self._aliases.get(
                (provider.casefold(), normalized_name(raw_name)), []
            )
            if alias.valid_on(when)
        ]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous aliases for {provider}:{raw_name!r} on {when}")
        if candidates:
            return Resolution(
                raw_name,
                candidates[0].team_id,
                ResolutionStatus.PROVIDER_ALIAS,
                provider,
            )
        team_id = self._canonical.get(normalized_name(raw_name))
        if team_id:
            return Resolution(raw_name, team_id, ResolutionStatus.EXACT_CANONICAL, provider)
        return Resolution(raw_name, None, ResolutionStatus.UNRESOLVED, provider)

    def resolve_new_team(
        self, provider: str, raw_name: str, when: date, *, approved_team_id: str
    ) -> Resolution:
        """Return an explicit cold-start resolution for a reviewed new club."""
        existing = self.resolve(provider, raw_name, when)
        if existing.team_id is not None:
            return existing
        if approved_team_id not in self._teams:
            raise ValueError(f"Approved new team is not in registry: {approved_team_id!r}")
        return Resolution(raw_name, approved_team_id, ResolutionStatus.NEW_TEAM_PRIOR, provider)


def assert_active_team_coverage(resolutions: list[Resolution]) -> None:
    unresolved = sorted({item.raw_name for item in resolutions if item.team_id is None})
    if unresolved:
        raise RuntimeError("Unresolved active team aliases: " + ", ".join(unresolved))
