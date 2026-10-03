import numpy as np
import pandas as pd
import pytest

from pitch_oracle_core.evaluation.pitchapi_ablation import PromotionPolicy, paired_block_bootstrap, per_fixture_scores, evaluate_pitchapi_families


def sample():
    dates = pd.date_range("2025-01-01", periods=180, freq="D", tz="UTC")
    y = np.tile(["H", "D", "A"], 60)
    return pd.DataFrame({"fixture_id": [f"fixture-{i}" for i in range(len(dates))], "league_key": "epl", "MatchDate": dates.strftime("%Y-%m-%d"), "kickoff_utc": dates + pd.Timedelta(hours=15), "as_of": dates + pd.Timedelta(hours=14), "feature_observed_at": dates - pd.Timedelta(days=1), "HomeTeam": "A", "AwayTeam": "B", "FullTimeResult": y, "EloDiff": np.random.default_rng(5).normal(size=len(dates)), "home_xt_for_ewm10": np.tile([2., 1., .1], 60)})


def policy():
    return PromotionPolicy(minimum_paired_fixtures=20, minimum_week_blocks=3, minimum_train_rows=30, bootstrap_repetitions=100, confidence=.95)


def test_paired_bootstrap_keeps_week_dependence_and_reproduces():
    base = np.array([1., 2., 3., 4., 5., 6.])
    candidate = base * .8
    blocks = np.array(["week1", "week1", "week2", "week2", "week3", "week3"])
    result = paired_block_bootstrap(base, candidate, blocks, repetitions=100)
    assert result == paired_block_bootstrap(base, candidate, blocks, repetitions=100)
    assert result["week_blocks"] == 3
    assert result["relative_improvement"] == pytest.approx(.2)
    assert result["lower"] == pytest.approx(.2)
    with pytest.raises(ValueError, match="paired"):
        paired_block_bootstrap(base, candidate[:-1], blocks)


def test_probabilities_and_outcomes_must_be_valid():
    ll, brier = per_fixture_scores([0], [[.8, .1, .1]])
    assert ll[0] == pytest.approx(-np.log(.8))
    assert brier[0] == pytest.approx(.06)
    with pytest.raises(ValueError, match="Invalid"):
        per_fixture_scores([0], [[.8, .1, .2]])


def test_walk_forward_reports_paired_folds_and_blocks_promotion_without_mapping():
    report, predictions = evaluate_pitchapi_families(sample(), league_key="epl", n_splits=2, policy=policy(), selected_families=("advanced_team",))
    assert set(report["ablations"]) == {f"A{i}" for i in range(10)}
    advanced = report["ablations"]["A3"]
    assert advanced["status"] == "evaluated"
    assert advanced["paired_fixtures"] == advanced["expected_test_fixtures"]
    assert not advanced["criteria"]["fixture_mapping"]
    assert not report["release_gate"]["promotion_passed"]
    for fold in advanced["folds"]:
        assert pd.Timestamp(fold["fit_through"]) < pd.Timestamp(fold["test_from"])
        assert pd.Timestamp(fold["calibration_through"]) < pd.Timestamp(fold["test_from"])
    assert not predictions.duplicated(["ablation", "fixture_id", "as_of"]).any()
    assert report["ablations"]["A6"]["status"] == "insufficient_data"


def test_late_provider_observation_fails_strict_audit_and_retrospective_cannot_promote():
    late = sample()
    late["feature_observed_at"] = late.kickoff_utc + pd.Timedelta(days=100)
    with pytest.raises(ValueError, match="cutoff"):
        evaluate_pitchapi_families(late, league_key="epl", n_splits=2, policy=policy())
    report, _ = evaluate_pitchapi_families(late, league_key="epl", n_splits=2, policy=policy(), strict=False, mapping_gate=True, selected_families=("advanced_team",))
    assert report["evaluation_mode"] == "retrospective"
    assert not report["ablations"]["A3"]["criteria"]["strict_observation_replay"]
    assert not report["release_gate"]["promotion_passed"]


def test_coverage_and_calibration_gates_report_missing_rows():
    data = sample()
    data.loc[data.index % 2 == 0, "home_xt_for_ewm10"] = np.nan
    report, _ = evaluate_pitchapi_families(data, league_key="epl", n_splits=2, policy=policy(), mapping_gate=True)
    record = report["ablations"]["A3"]
    assert record["coverage"] == pytest.approx(.5)
    assert not record["criteria"]["coverage"]
    assert "calibration_error" in record["baseline"]
    assert any(group["dimension"] == "result_class" for group in record["subgroups"])


def test_stages_and_leagues_cannot_be_mixed():
    mixed = sample().assign(forecast_stage="initial")
    mixed.loc[1, "forecast_stage"] = "lineup"
    with pytest.raises(ValueError, match="stages separately"):
        evaluate_pitchapi_families(mixed, league_key="epl", policy=policy())
    with pytest.raises(ValueError, match="different league"):
        evaluate_pitchapi_families(sample(), league_key="eredivisie", policy=policy())
