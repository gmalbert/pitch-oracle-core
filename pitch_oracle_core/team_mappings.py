"""Centralised team-name normalisation shared by every scraper and ingest script.

Every ``penaltyblog`` scraper accepts a ``team_mappings`` dict in the form
``{canonical_name: [provider_alias, ...]}``. This module builds that dict once
from the league configurations so all fetch scripts reconcile team identities
the same way — including Belgium and the Bundesliga 2, which the legacy
root-level ``team_name_mapping.py`` did not cover.

Canonical names follow the football-data.co.uk historical convention (short
display names such as ``"Man City"`` and ``"Nott'm Forest"``); aliases are the
spellings used by FBRef, Understat, ClubElo, ESPN, and API-Football.
"""

from __future__ import annotations

from collections.abc import Mapping

from penaltyblog.scrapers import get_example_team_name_mappings
from penaltyblog.scrapers.common import COMPETITION_MAPPINGS

from .config import LeagueConfig

# ---------------------------------------------------------------------------
# Source maps (provider spelling -> canonical display name, unless noted).
# These were previously duplicated across team_name_mapping.py,
# fetch_understat_xg.py, and fetch_clubelo.py; they now live here exactly once.
# ---------------------------------------------------------------------------

# ESPN / API-Football full names -> canonical display name (EPL).
TEAM_NAME_MAP: dict[str, str] = {
    "Manchester United": "Man United", "Manchester City": "Man City",
    "Wolverhampton Wanderers": "Wolves", "Brighton & Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'm Forest", "AFC Bournemouth": "Bournemouth",
    "Newcastle United": "Newcastle", "West Ham United": "West Ham",
    "Tottenham Hotspur": "Tottenham", "Leeds United": "Leeds",
}

# Understat display name -> canonical display name (EPL).
UNDERSTAT_TEAM_MAP: dict[str, str] = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Arsenal": "Arsenal", "Chelsea": "Chelsea", "Liverpool": "Liverpool",
    "Tottenham": "Tottenham", "Everton": "Everton", "Aston Villa": "Aston Villa",
    "Newcastle United": "Newcastle", "West Ham": "West Ham",
    "Crystal Palace": "Crystal Palace", "Southampton": "Southampton",
    "Leicester": "Leicester", "Fulham": "Fulham", "Brentford": "Brentford",
    "Brighton": "Brighton", "Bournemouth": "Bournemouth",
    "Wolverhampton Wanderers": "Wolves", "Nottingham Forest": "Nott'm Forest",
    "Burnley": "Burnley", "Leeds": "Leeds", "Watford": "Watford",
    "Norwich": "Norwich", "Sheffield Utd": "Sheffield United",
    "Sunderland": "Sunderland", "Stoke": "Stoke", "Swansea": "Swansea",
    "Hull": "Hull", "Reading": "Reading", "QPR": "QPR", "West Brom": "West Brom",
    "Middlesbrough": "Middlesbrough", "Blackburn": "Blackburn",
    "Huddersfield": "Huddersfield", "Cardiff": "Cardiff", "Luton": "Luton",
    "Ipswich": "Ipswich",
}

# Canonical display name -> ClubElo URL slug (verify at clubelo.com).
CLUBELO_TEAM_MAP: dict[str, str] = {
    "Man City": "ManCity", "Man United": "ManUnited", "Arsenal": "Arsenal",
    "Chelsea": "Chelsea", "Liverpool": "Liverpool", "Tottenham": "Tottenham",
    "Everton": "Everton", "Aston Villa": "AstonVilla", "Newcastle": "Newcastle",
    "West Ham": "WestHam", "Crystal Palace": "CrystalPalace",
    "Southampton": "Southampton", "Leicester": "Leicester", "Fulham": "Fulham",
    "Brentford": "Brentford", "Brighton": "Brighton", "Bournemouth": "Bournemouth",
    "Wolves": "Wolves", "Nott'm Forest": "Forest", "Burnley": "Burnley",
    "Leeds": "Leeds", "Watford": "Watford", "Norwich": "Norwich",
    "Sheffield United": "SheffieldUnited", "Sunderland": "Sunderland",
    "Stoke": "Stoke", "Swansea": "Swansea", "Hull": "Hull", "Reading": "Reading",
    "QPR": "QPR", "West Brom": "WestBrom", "Middlesbrough": "Middlesbrough",
    "Blackburn": "Blackburn", "Wigan": "Wigan", "Bolton": "Bolton",
    "Ipswich": "Ipswich", "Derby": "Derby", "Birmingham": "Birmingham",
    "Huddersfield": "Huddersfield", "Cardiff": "Cardiff",
    "Sheffield Weds": "SheffieldWeds", "Charlton": "Charlton", "Luton": "Luton",
}


def _invert(source: Mapping[str, str]) -> dict[str, list[str]]:
    """Invert ``alias -> canonical`` into the penaltyblog ``canonical -> [aliases]`` form."""
    inverted: dict[str, list[str]] = {}
    for alias, canonical in source.items():
        if alias == canonical:
            continue
        inverted.setdefault(canonical, [])
        if alias not in inverted[canonical]:
            inverted[canonical].append(alias)
    return inverted


def _merge(target: dict[str, list[str]], source: Mapping[str, str]) -> None:
    for canonical, aliases in _invert(source).items():
        existing = target.setdefault(canonical, [])
        for alias in aliases:
            if alias not in existing:
                existing.append(alias)


def _build_canonical_mappings() -> dict[str, list[str]]:
    from .leagues import BUILTIN_LEAGUES

    merged: dict[str, list[str]] = _invert(TEAM_NAME_MAP)
    _merge(merged, UNDERSTAT_TEAM_MAP)
    for canonical, slug in CLUBELO_TEAM_MAP.items():
        if slug != canonical:
            existing = merged.setdefault(canonical, [])
            if slug not in existing:
                existing.append(slug)
    for config in BUILTIN_LEAGUES.values():
        _merge(merged, config.team_aliases)
    return merged


# penaltyblog-format mappings (canonical -> [aliases]) covering every builtin
# league. Pass straight to ``ClubElo`` / ``Understat`` / ``FootballData`` /
# ``FBRef`` as ``team_mappings=CANONICAL_TEAM_MAPPINGS``.
CANONICAL_TEAM_MAPPINGS: dict[str, list[str]] = _build_canonical_mappings()

# Re-export the upstream examples so any consumer repo has a sensible starting
# point for leagues the builtin configs do not cover.
EXAMPLE_MAPPINGS: dict[str, list[str]] = get_example_team_name_mappings()


def mappings_for_league(league: LeagueConfig | str) -> dict[str, list[str]]:
    """Return penaltyblog-format team mappings scoped to one league.

    Consumer-supplied leagues extend the canonical dict with their own
    ``team_aliases`` so the scrapers normalise identities consistently even
    outside the builtin set.
    """
    from .leagues import get_league_config

    config = get_league_config(league) if isinstance(league, str) else league
    merged = {canonical: list(aliases) for canonical, aliases in CANONICAL_TEAM_MAPPINGS.items()}
    _merge(merged, config.team_aliases)
    return merged


def display_name_map(league: LeagueConfig | str) -> dict[str, str]:
    """Return ``alias -> canonical`` for one league (the normalisation direction)."""
    mapping: dict[str, str] = {}
    for canonical, aliases in mappings_for_league(league).items():
        mapping.setdefault(canonical, canonical)
        for alias in aliases:
            mapping[alias] = canonical
    return mapping


def normalize_team_name(team_name: str, aliases: Mapping[str, str] | None = None) -> str:
    """Map a provider team name onto its canonical display name.

    Backwards-compatible with the deleted root-level ``team_name_mapping.py``:
    the EPL map is the default and caller-supplied aliases take precedence.
    """
    return dict(TEAM_NAME_MAP, **(aliases or {})).get(team_name, team_name)


def penaltyblog_competition(league: LeagueConfig | str, source: str = "footballdata") -> str:
    """Resolve the penaltyblog competition name for a league and data source.

    ``source`` is a key in ``penaltyblog.scrapers.common.COMPETITION_MAPPINGS``
    (``"footballdata"``, ``"understat"``, or ``"fbref"``). The lookup uses the
    league's football-data division code or Understat slug, so no per-league
    competition strings are duplicated here.
    """
    from .leagues import get_league_config

    config = get_league_config(league) if isinstance(league, str) else league
    if source == "footballdata":
        slug = config.football_data_div
    elif source == "understat":
        slug = config.sources.understat_league
    else:
        slug = None
    if not slug:
        raise ValueError(f"No {source} slug configured for {config.key}")
    for competition, sources in COMPETITION_MAPPINGS.items():
        entry = sources.get(source)
        if entry and entry.get("slug") == slug:
            return competition
    raise ValueError(f"No penaltyblog competition maps {source} slug {slug!r} ({config.key})")
