import pytest
from dataclasses import replace

from pitch_oracle_core import (
    OptionalFeatureSet, Shot, apply_phase_transition, assign_phase,
    eligible_opponents, expected_goals_from_shots, get_league_config,
    normalize_odds,
)
from pitch_oracle_core.providers import ProviderRegistry
from pitch_oracle_core.training import TrainingResult
from pitch_oracle_core.runtime import Runtime


def test_all_reference_leagues_have_identifiers():
    for key in ("epl", "scotland", "eredivisie", "portugal", "belgium", "turkey"):
        config = get_league_config(key)
        assert config.football_data_div
        assert config.team_count > 0


def test_scotland_split_and_opponents():
    config = get_league_config("scotland")
    assert assign_phase(33, config) == "regular"
    assert assign_phase(34, config) == "split"
    assert eligible_opponents("A", "split", {"top_6": ["A", "B"]}) == {"B"}


def test_belgium_halves_and_rounds_up_points():
    config = get_league_config("belgium")
    assert apply_phase_transition({"A": 51, "B": 50}, config) == {"A": 26, "B": 25}


def test_shot_xg_is_deterministic_and_empty_is_safe():
    shots = [Shot(distance=10), Shot(distance=20, header=True)]
    assert expected_goals_from_shots(shots) == pytest.approx(1.0, abs=0.01)
    assert expected_goals_from_shots([]) == 0


def test_optional_sources_do_not_create_missing_features():
    assert OptionalFeatureSet().as_model_features() == {}
    assert OptionalFeatureSet(injuries={"absences": 2}).as_model_features() == {"injuries_absences": 2}


def test_odds_normalization_and_validation():
    event = normalize_odds({
        "event_id": 1, "home_team": "A", "away_team": "B",
        "markets": [{"outcome": "home", "decimal_price": "2.1"}],
    }, provider="free-provider")
    assert event.provider == "free-provider"
    assert event.markets[0].decimal_price == 2.1
    with pytest.raises(ValueError):
        normalize_odds({"event_id": 1, "home_team": "A", "away_team": "B",
                        "markets": [{"outcome": "home", "decimal_price": 1}]}, provider="x")


def test_epl_provider_configuration_is_parameterized():
    config = get_league_config("epl")
    assert config.sources.api_football_league_id == 39
    assert config.sources.understat_league == "EPL"
    assert config.team_aliases["Manchester City"] == "Man City"
    assert "Anfield" not in config.stadium_coordinates
    assert config.stadium_coordinates["Liverpool"] == (53.4308, -2.9608)


def test_provider_registry_skips_unavailable_optional_features():
    assert ProviderRegistry().fetch_optional_features(get_league_config("eredivisie")) == {}
    assert TrainingResult.__annotations__["league"] is str


def test_runtime_uses_consumer_configured_directories(tmp_path):
    config = replace(
        get_league_config("eredivisie"),
        data_dir_name="league_data",
        models_dir_name="league_models",
    )
    runtime = Runtime.for_league(config, tmp_path)

    assert runtime.data_dir == tmp_path / "league_data"
    assert runtime.models_dir == tmp_path / "league_models"
    assert runtime.environment()["PITCH_ORACLE_LEAGUE"] == "eredivisie"
