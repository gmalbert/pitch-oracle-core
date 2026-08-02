from datetime import date

from combine_raw_data import recent_season_codes
from pitch_oracle_core import get_league_config


def test_recent_seasons_advance_without_a_hardcoded_epl_year():
    assert recent_season_codes(date(2026, 8, 2), count=5) == (
        "2223", "2324", "2425", "2526", "2627"
    )
    assert recent_season_codes(date(2027, 2, 1), count=2) == ("2526", "2627")


def test_pilot_consumer_has_no_required_epl_optional_sources():
    config = get_league_config("eredivisie")
    assert config.football_data_div == "N1"
    assert not config.sources.injuries
    assert not config.sources.referee
    assert not config.sources.understat
