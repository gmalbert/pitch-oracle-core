from config import LEAGUE_CONFIG


def test_consumer_selects_a_registered_non_epl_league():
    assert LEAGUE_CONFIG.key == "eredivisie"
    assert LEAGUE_CONFIG.football_data_div == "N1"
    assert LEAGUE_CONFIG.display_name == "Eredivisie"
