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
_EREDIVISIE_SOURCES = DataSourceConfig(
    understat=False,
    referee=False,
    injuries=False,
    weather_timezone="Europe/Amsterdam",
)
_EREDIVISIE_TEAM_ALIASES = {
    "Ajax Amsterdam": "Ajax",
    "FC Groningen": "Groningen",
    "FC Twente": "Twente",
    "FC Utrecht": "Utrecht",
    "Feyenoord Rotterdam": "Feyenoord",
    "Fortuna Sittard": "For Sittard",
    "NEC Nijmegen": "Nijmegen",
    "PEC Zwolle": "Zwolle",
    "SC Cambuur": "Cambuur",
}
_EREDIVISIE_STADIUMS = {
    # Current and recent Eredivisie clubs. Historical aliases are included so
    # the weather backfill can cover the full data set, not only current fixtures.
    "Ajax": (52.3140, 4.9414),
    "ADO Den Haag": (52.0628, 4.3832),
    "AZ Alkmaar": (52.6125, 4.7415),
    "Almere City": (52.3505, 5.2647),
    "Cambuur": (53.2013, 5.7998),
    "SC Cambuur": (53.2013, 5.7998),
    "Excelsior": (51.9170, 4.5355),
    "FC Emmen": (52.7571, 6.9716),
    "Feyenoord": (51.8939, 4.5231),
    "For Sittard": (50.9996, 5.8161),
    "Fortuna Sittard": (50.9996, 5.8161),
    "Go Ahead Eagles": (52.2554, 6.1639),
    "Groningen": (53.2044, 6.5665),
    "FC Groningen": (53.2044, 6.5665),
    "Heerenveen": (52.9579, 5.9327),
    "Heracles": (52.3360, 6.6537),
    "NAC Breda": (51.5964, 4.7496),
    "Nijmegen": (51.8203, 5.8372),
    "NEC Nijmegen": (51.8203, 5.8372),
    "PSV Eindhoven": (51.4416, 5.4676),
    "Sparta Rotterdam": (51.9194, 4.4330),
    "Telstar": (52.4575, 4.6555),
    "Twente": (52.2363, 6.8350),
    "FC Twente": (52.2363, 6.8350),
    "Utrecht": (52.0784, 5.1456),
    "Vitesse": (51.9637, 5.8880),
    "Volendam": (52.4887, 5.0586),
    "Waalwijk": (51.6855, 5.0705),
    "Willem II": (51.5425, 5.0663),
    "Zwolle": (52.5048, 6.0881),
}
_TURKEY_TEAM_ALIASES = {
    "Caykur Rizespor": "Rizespor",
    "Gaziantep FK": "Gaziantep",
    "Goztepe": "Goztep",
    "Istanbul Basaksehir": "Buyuksehyr",
}

BUILTIN_LEAGUES = {
    "epl": LeagueConfig("epl", "Premier League", "E0", "eng.1", "ENG_1", 20, (8, 5),
                         country_name="England", country_flag="🇬🇧",
                         team_aliases=_EPL_TEAM_ALIASES, stadium_coordinates=_EPL_STADIUMS,
                         sources=_EPL_SOURCES),
    "scotland": LeagueConfig(
        "scotland", "Scottish Premiership", "SC0", "sco.1", "SCO_1", 12, (7, 5),
        country_name="Scotland", country_flag="🇬🇧",
        phase=PhaseConfig(regular_matches_per_opponent=3, split_after_round=33,
                          split_pools=("top_6", "bottom_6"), split_pool_sizes=(6, 6), playoffs=(
                              PlayoffConfig("promotion_relegation", cross_division=True),)),
        sources=_OPTIONAL,
    ),
    "eredivisie": LeagueConfig(
        "eredivisie", "Eredivisie", "N1", "ned.1", "NED_1", 18, (8, 5),
        country_name="Netherlands", country_flag="🇳🇱",
        phase=PhaseConfig(playoffs=(PlayoffConfig("european_qualification"),)),
        team_aliases=_EREDIVISIE_TEAM_ALIASES,
        stadium_coordinates=_EREDIVISIE_STADIUMS, sources=_EREDIVISIE_SOURCES,
    ),
    "portugal": LeagueConfig("portugal", "Primeira Liga", "P1", None, "POR_1", 18, (8, 5),
                              country_name="Portugal", country_flag="🇵🇹", sources=_OPTIONAL),
    "belgium": LeagueConfig(
        "belgium", "Belgian Pro League", "B1", "bel.1", "BEL_1", 18, (8, 5),
        country_name="Belgium", country_flag="🇧🇪",
        phase=PhaseConfig(split_after_round=34, split_pools=("champions", "europe", "relegation"),
                          split_pool_sizes=(6, 6, 6),
                          points_halving=True, points_halving_rounding="ceil"), sources=_OPTIONAL,
    ),
    "turkey": LeagueConfig(
        "turkey", "Süper Lig", "T1", "tur.1", "TUR_1", 20, (8, 5),
        country_name="Turkey", country_flag="🇹🇷",
        points_adjustments={}, team_aliases=_TURKEY_TEAM_ALIASES, sources=_OPTIONAL,
    ),
}


def get_league_config(key: str) -> LeagueConfig:
    try:
        return BUILTIN_LEAGUES[key.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown league {key!r}; choose from {sorted(BUILTIN_LEAGUES)}") from exc
