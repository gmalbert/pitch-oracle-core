from pathlib import Path

from config import LEAGUE_CONFIG
from pitch_oracle_core import __version__


ROOT = Path(__file__).resolve().parents[1]
CORE_REF = "v1.3.22"


def test_consumer_selects_a_registered_non_epl_league():
    assert LEAGUE_CONFIG.key == "eredivisie"
    assert LEAGUE_CONFIG.key != "epl"
    assert LEAGUE_CONFIG.football_data_div
    assert LEAGUE_CONFIG.espn_slug


def test_core_pin_is_synchronized_everywhere():
    assert __version__ == CORE_REF.removeprefix("v")
    pin = f"pitch-oracle-core[consumer] @ git+https://github.com/gmalbert/pitch-oracle-core.git@{CORE_REF}"
    assert pin in (ROOT / "requirements.txt").read_text()
    assert pin in (ROOT / "requirements-ci.txt").read_text()
    workflow = (ROOT / ".github" / "workflows" / "artifact-pipeline.yml").read_text()
    assert f"precompute-consumer.yml@{CORE_REF}" in workflow
    assert f"core_ref: {CORE_REF}" in workflow
