"""Reference configurations for the initial European league consumers."""

from .config import DataSourceConfig, LeagueConfig, PhaseConfig, PlayoffConfig

_EPL_SOURCES = DataSourceConfig(
    understat=True, understat_league="EPL", referee=True, injuries=True,
    api_football_league_id=39, live_odds_providers=("bzzoiro",),
)
_OPTIONAL = DataSourceConfig(understat=False, referee=False, injuries=False)
_EPL_TEAM_ALIASES = {
    "Manchester United": "Man United", "Manchester City": "Man City",
    "Wolverhampton Wanderers": "Wolves", "Brighton & Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'm Forest", "AFC Bournemouth": "Bournemouth",
    "Newcastle United": "Newcastle", "West Ham United": "West Ham",
    "Tottenham Hotspur": "Tottenham", "Leeds United": "Leeds",
}
_EPL_STADIUMS = {
    "Man United": (53.4631, -2.2913), "Arsenal": (51.5549, -0.1084),
    "Liverpool": (53.4308, -2.9608), "Chelsea": (51.4817, -0.1910),
    "Tottenham": (51.6043, -0.0664), "Man City": (53.4831, -2.2004),
    "Aston Villa": (52.5093, -1.8848), "Everton": (53.4389, -2.9663),
    "Newcastle": (54.9756, -1.6217), "Crystal Palace": (51.3983, -0.0855),
    "West Ham": (51.5383, -0.0164), "Wolves": (52.5903, -2.1302),
    "Brighton": (50.8618, -0.0833), "Brentford": (51.4908, -0.2887),
    "Bournemouth": (50.7352, -1.8383), "Fulham": (51.4750, -0.2217),
    "Leeds": (53.7778, -1.5722), "Nott'm Forest": (52.9399, -1.1326),
}

BUILTIN_LEAGUES = {
    "epl": LeagueConfig("epl", "Premier League", "E0", "eng.1", "ENG_1", 20, (8, 5),
                         team_aliases=_EPL_TEAM_ALIASES, stadium_coordinates=_EPL_STADIUMS,
                         sources=_EPL_SOURCES),
    "scotland": LeagueConfig(
        "scotland", "Scottish Premiership", "SC0", "sco.1", "SCO_1", 12, (7, 5),
        phase=PhaseConfig(regular_matches_per_opponent=3, split_after_round=33,
                          split_pools=("top_6", "bottom_6"), playoffs=(
                              PlayoffConfig("promotion_relegation", cross_division=True),)),
        sources=_OPTIONAL,
    ),
    "eredivisie": LeagueConfig(
        "eredivisie", "Eredivisie", "N1", "ned.1", "NED_1", 18, (8, 5),
        phase=PhaseConfig(playoffs=(PlayoffConfig("european_qualification"),)), sources=_OPTIONAL,
    ),
    "portugal": LeagueConfig("portugal", "Primeira Liga", "P1", None, "POR_1", 18, (8, 5), sources=_OPTIONAL),
    "belgium": LeagueConfig(
        "belgium", "Belgian Pro League", "B1", "bel.1", "BEL_1", 18, (8, 5),
        phase=PhaseConfig(split_after_round=34, split_pools=("champions", "europe", "relegation"),
                          points_halving=True, points_halving_rounding="ceil"), sources=_OPTIONAL,
    ),
    "turkey": LeagueConfig(
        "turkey", "Süper Lig", "T1", None, "TUR_1", 20, (8, 5),
        points_adjustments={}, sources=_OPTIONAL,
    ),
}


def get_league_config(key: str) -> LeagueConfig:
    try:
        return BUILTIN_LEAGUES[key.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown league {key!r}; choose from {sorted(BUILTIN_LEAGUES)}") from exc
