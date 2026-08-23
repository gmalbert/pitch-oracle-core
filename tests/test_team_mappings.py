"""Tests for pitch_oracle_core.team_mappings — the shared normalisation registry."""

import pytest

from pitch_oracle_core.team_mappings import (
    CANONICAL_TEAM_MAPPINGS,
    CLUBELO_TEAM_MAP,
    TEAM_NAME_MAP,
    UNDERSTAT_TEAM_MAP,
    display_name_map,
    mappings_for_league,
    normalize_team_name,
    penaltyblog_competition,
)


class TestCanonicalMappings:
    def test_epl_canonical_names_have_both_provider_aliases(self):
        assert "Manchester City" in CANONICAL_TEAM_MAPPINGS["Man City"]
        assert "ManCity" in CANONICAL_TEAM_MAPPINGS["Man City"]

    def test_understat_aliases_are_merged(self):
        assert "Sheffield Utd" in CANONICAL_TEAM_MAPPINGS["Sheffield United"]

    def test_belgium_aliases_are_merged(self):
        assert "Racing Genk" in CANONICAL_TEAM_MAPPINGS["Genk"]
        assert "Union St.-Gilloise" in CANONICAL_TEAM_MAPPINGS["St. Gilloise"]

    def test_eredivisie_aliases_are_merged(self):
        assert "Ajax Amsterdam" in CANONICAL_TEAM_MAPPINGS["Ajax"]

    def test_portugal_aliases_are_merged(self):
        assert "Vit\u00f3ria SC" in CANONICAL_TEAM_MAPPINGS["Guimaraes"] or \
               "Vitoria SC" in CANONICAL_TEAM_MAPPINGS["Guimaraes"]

    def test_turkey_aliases_are_merged(self):
        assert "Caykur Rizespor" in CANONICAL_TEAM_MAPPINGS["Rizespor"]

    def test_no_canonical_name_appears_as_own_alias(self):
        for canonical, aliases in CANONICAL_TEAM_MAPPINGS.items():
            assert canonical not in aliases


class TestNormalizeTeamName:
    def test_legacy_epl_map(self):
        assert normalize_team_name("Manchester United") == "Man United"
        assert normalize_team_name("Wolverhampton Wanderers") == "Wolves"

    def test_unknown_name_passes_through(self):
        assert normalize_team_name("Some Unknown FC") == "Some Unknown FC"

    def test_caller_aliases_take_precedence(self):
        result = normalize_team_name("Custom FC", {"Custom FC": "Custom"})
        assert result == "Custom"


class TestMappingsForLeague:
    def test_epl_includes_global_and_league_aliases(self):
        m = mappings_for_league("epl")
        assert "Manchester City" in m["Man City"]
        assert "ManCity" in m["Man City"]

    def test_belgium_includes_league_aliases(self):
        m = mappings_for_league("belgium")
        assert "Racing Genk" in m["Genk"]

    def test_league_config_object_accepted(self):
        from pitch_oracle_core.leagues import get_league_config
        m = mappings_for_league(get_league_config("turkey"))
        assert "Caykur Rizespor" in m["Rizespor"]


class TestDisplayNameMap:
    def test_bidirectional_epl(self):
        m = display_name_map("epl")
        assert m["Manchester United"] == "Man United"
        assert m["Man United"] == "Man United"

    def test_belgium_bidirectional(self):
        m = display_name_map("belgium")
        assert m["Racing Genk"] == "Genk"
        assert m["Genk"] == "Genk"


class TestPenaltyblogCompetition:
    @pytest.mark.parametrize("league,expected", [
        ("epl", "ENG Premier League"),
        ("belgium", "BEL First Division A"),
        ("scotland", "SCO Premier League"),
        ("eredivisie", "NLD Eredivisie"),
        ("portugal", "PRT Liga 1"),
        ("turkey", "TUR Super Lig"),
    ])
    def test_footballdata_competitions(self, league, expected):
        assert penaltyblog_competition(league, "footballdata") == expected

    def test_understat_epl(self):
        assert penaltyblog_competition("epl", "understat") == "ENG Premier League"

    def test_understat_unavailable_raises(self):
        with pytest.raises(ValueError, match="No understat slug"):
            penaltyblog_competition("belgium", "understat")

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="No .* slug configured"):
            penaltyblog_competition("epl", "nonexistent")
