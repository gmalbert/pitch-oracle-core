from pathlib import Path

from config import LEAGUE_CONFIG
from pitch_oracle_core import __version__


ROOT = Path(__file__).resolve().parents[1]
CORE_REF = "6332908ffe29a104de2dfc6b1209b4c896c6bca0"


def test_consumer_selects_a_registered_non_epl_league():
    assert LEAGUE_CONFIG.key == "eredivisie"
    assert LEAGUE_CONFIG.key != "epl"
    assert LEAGUE_CONFIG.football_data_div
    assert LEAGUE_CONFIG.espn_slug


def test_core_pin_is_synchronized_everywhere():
    assert __version__ == CORE_REF.removeprefix("v")
    runtime_pin = f"pitch-oracle-core[runtime] @ git+https://github.com/gmalbert/pitch-oracle-core.git@{CORE_REF}"
    pipeline_pin = f"pitch-oracle-core[pipeline,diagnostics] @ git+https://github.com/gmalbert/pitch-oracle-core.git@{CORE_REF}"
    assert runtime_pin in (ROOT / "requirements.txt").read_text()
    assert pipeline_pin in (ROOT / "requirements-ci.txt").read_text()
    workflow = (ROOT / ".github" / "workflows" / "artifact-pipeline.yml").read_text()
    assert f"precompute-consumer.yml@{CORE_REF}" in workflow
    assert f"core_ref: {CORE_REF}" in workflow
