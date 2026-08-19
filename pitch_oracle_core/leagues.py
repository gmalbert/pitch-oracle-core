"""Reference configurations for the initial European league consumers."""

from .config import DataSourceConfig, LeagueConfig, PhaseConfig, PlayoffConfig

_EPL_SOURCES = DataSourceConfig(
    understat=True, understat_league="EPL", referee=True, injuries=True,
    api_football_league_id=39, live_odds_providers=("bzzoiro",),
    pitchapi=True, pitchapi_league_id="l_4WFCIZ",
)
_BELGIUM_SOURCES = DataSourceConfig(
    understat=False,
    referee=False,
    injuries=False,
    weather_timezone="Europe/Brussels",
    pitchapi=True, pitchapi_league_id="l_2L6d1F",
)
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
    pitchapi=True, pitchapi_league_id="l_4H43wr",
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
_BELGIUM_TEAM_ALIASES = {
    "Cercle Brugge KSV": "Cercle Brugge",
    "KAA Gent": "Gent",
    "KV Kortrijk": "Kortrijk",
    "KV Mechelen": "Mechelen",
    "KVC Westerlo": "Westerlo",
    "OH Leuven": "Leuven",
    "RAAL La Louvière": "RAAL La Louviere",
    "Racing Genk": "Genk",
    "Royal Charleroi SC": "Charleroi",
    "Sint-Truidense": "St. Truiden",
    "Standard Liege": "Standard",
    "Union St.-Gilloise": "St. Gilloise",
    "Waasland-Beveren": "Beveren",
    "Zulte-Waregem": "Waregem",
}
_PORTUGAL_STADIUMS = {
    "AVS": (41.3900, -8.4100),
    "Academico Viseu": (40.6580, -7.9100),
    "Alverca": (38.8940, -9.0380),
    "Arouca": (40.9300, -8.2500),
    "Benfica": (38.7516, -9.1841),
    "Boavista": (41.1616, -8.6600),
    "Casa Pia": (38.7380, -9.1620),
    "Chaves": (41.7400, -7.4700),
    "Estoril": (38.7030, -9.4000),
    "Estrela": (38.7500, -9.2300),
    "Famalicao": (41.4100, -8.5200),
    "Farense": (37.0190, -7.9340),
    "Gil Vicente": (41.5340, -8.6200),
    "Guimaraes": (41.4440, -8.3000),
    "Maritimo": (32.6500, -16.9100),
    "Moreirense": (41.3800, -8.3600),
    "Nacional": (32.6680, -16.9200),
    "Pacos Ferreira": (41.2800, -8.3800),
    "Portimonense": (37.1400, -8.5400),
    "Porto": (41.1618, -8.5832),
    "Rio Ave": (41.3500, -8.7500),
    "Santa Clara": (37.7400, -25.6600),
    "Sp Braga": (41.5500, -8.4300),
    "Sp Lisbon": (38.7567, -9.1658),
    "Tondela": (40.5200, -8.0800),
    "Vizela": (41.3900, -8.3100),
}

BUILTIN_LEAGUES = {
    "epl": LeagueConfig("epl", "Premier League", "E0", "eng.1", "ENG_1", 20, (8, 5),
                         country_name="England", country_flag="🇬🇧",
                         team_aliases=_EPL_TEAM_ALIASES, stadium_coordinates=_EPL_STADIUMS,
                         sources=_EPL_SOURCES,
                         outcome_labels={"title": (1,), "champions_league": (1, 2, 3, 4), "relegation": (18, 19, 20)}),
    "scotland": LeagueConfig(
        "scotland", "Scottish Premiership", "SC0", "sco.1", "SCO_1", 12, (7, 5),
        country_name="Scotland", country_flag="🇬🇧",
        phase=PhaseConfig(regular_matches_per_opponent=3, split_after_round=33,
                          split_pools=("top_6", "bottom_6"), split_pool_sizes=(6, 6), playoffs=(
                              PlayoffConfig(
                                  "promotion_relegation", cross_division=True,
                                  sources=("premiership:11", "championship:playoff_winner"),
                                  legs=2, outcome_label="premiership_place",
                              ),)),
        sources=DataSourceConfig(
            understat=False, referee=False, injuries=False,
            pitchapi=True, pitchapi_league_id="l_1LMdEO",
        ),
        outcome_labels={"title": (1,), "top_six": (1, 2, 3, 4, 5, 6), "relegation_playoff": (11,), "relegation": (12,)},
    ),
    "eredivisie": LeagueConfig(
        "eredivisie", "Eredivisie", "N1", "ned.1", "NED_1", 18, (8, 5),
        country_name="Netherlands", country_flag="🇳🇱",
        phase=PhaseConfig(playoffs=(PlayoffConfig(
            "european_qualification",
            sources=("league:5", "league:6", "league:7", "league:8"),
            outcome_label="europe",
        ),)),
        team_aliases=_EREDIVISIE_TEAM_ALIASES,
        stadium_coordinates=_EREDIVISIE_STADIUMS, sources=_EREDIVISIE_SOURCES,
        outcome_labels={"title": (1,), "europe": (1, 2, 3, 4), "relegation": (17, 18)},
    ),
    "portugal": LeagueConfig("portugal", "Primeira Liga", "P1", None, "POR_1", 18, (8, 5),
                              country_name="Portugal", country_flag="🇵🇹",
                              stadium_coordinates=_PORTUGAL_STADIUMS,
                              sources=DataSourceConfig(
                                  understat=False, referee=False, injuries=False,
                                  weather_timezone="Europe/Lisbon",
                                  pitchapi=True, pitchapi_league_id="l_4QexZg",
                                  live_odds_providers=("bzzoiro",),
                              ),
                              outcome_labels={"title": (1,), "europe": (1, 2, 3, 4, 5), "relegation": (16, 17, 18)}),
    "belgium": LeagueConfig(
        "belgium", "Belgian Pro League", "B1", "bel.1", "BEL_1", 18, (8, 5),
        country_name="Belgium", country_flag="🇧🇪",
        phase=PhaseConfig(split_after_round=34, split_pools=("champions", "europe", "relegation"),
                          split_pool_sizes=(6, 6, 6),
                          points_halving=True, points_halving_rounding="ceil"),
        team_aliases=_BELGIUM_TEAM_ALIASES,
        sources=_BELGIUM_SOURCES,
        outcome_labels={"title": (1,), "champions_playoff": (1, 2, 3, 4, 5, 6), "europe_playoff": (7, 8, 9, 10, 11, 12), "relegation_playoff": (13, 14, 15, 16, 17, 18)},
    ),
    "turkey": LeagueConfig(
        "turkey", "Süper Lig", "T1", "tur.1", "TUR_1", 20, (8, 5),
        country_name="Turkey", country_flag="🇹🇷",
        points_adjustments={}, team_aliases=_TURKEY_TEAM_ALIASES,
        sources=DataSourceConfig(
            understat=False, referee=False, injuries=False,
            pitchapi=True, pitchapi_league_id="l_0S1uaf",
        ),
        outcome_labels={"title": (1,), "europe": (1, 2, 3, 4), "relegation": (17, 18, 19, 20)},
    ),
}


def get_league_config(key: str) -> LeagueConfig:
    try:
        return BUILTIN_LEAGUES[key.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown league {key!r}; choose from {sorted(BUILTIN_LEAGUES)}") from exc
