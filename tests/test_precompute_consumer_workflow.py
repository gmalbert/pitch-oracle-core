from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "precompute-consumer.yml"


def test_prepare_step_bypasses_loopback_proxy_for_football_data():
    text = WORKFLOW.read_text(encoding="utf-8")
    prepare_step = text.split("- name: Prepare league data", 1)[1].split("- name:", 1)[0]

    assert "NO_PROXY: football-data.co.uk,www.football-data.co.uk" in prepare_step
    assert "no_proxy: football-data.co.uk,www.football-data.co.uk" in prepare_step
