"""League-supplied team aliases with EPL-compatible default behavior."""

from collections.abc import Mapping

TEAM_NAME_MAP = {
    "Manchester United": "Man United", "Manchester City": "Man City",
    "Wolverhampton Wanderers": "Wolves", "Brighton & Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'm Forest", "AFC Bournemouth": "Bournemouth",
    "Newcastle United": "Newcastle", "West Ham United": "West Ham",
    "Tottenham Hotspur": "Tottenham", "Leeds United": "Leeds",
}


def normalize_team_name(team_name: str, aliases: Mapping[str, str] | None = None) -> str:
    return dict(TEAM_NAME_MAP, **(aliases or {})).get(team_name, team_name)

