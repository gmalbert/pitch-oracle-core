import pickle

import numpy as np
import pandas as pd
import pytest

from pitch_oracle_core.features import FEATURE_POLICY_VERSION
from pitch_oracle_core.predictions import (
    FeatureContract,
    build_prediction_frame,
    build_upcoming_feature_matrix,
)


def _contract():
    return FeatureContract(
        version=FEATURE_POLICY_VERSION,
        feature_names=("HomeMomentum_L3", "AwayMomentum_L3", "MarketMargin"),
        imputation_values={
            "HomeMomentum_L3": 3.0,
            "AwayMomentum_L3": 3.0,
            "MarketMargin": 0.05,
        },
        state_sources={
            "HomeMomentum_L3": {
                "fixture_role": "home",
                "home_history_column": "HomeMomentum_L3",
                "away_history_column": "AwayMomentum_L3",
            },
            "AwayMomentum_L3": {
                "fixture_role": "away",
                "home_history_column": "HomeMomentum_L3",
                "away_history_column": "AwayMomentum_L3",
            },
        },
    )


def test_feature_contract_loads_from_precomputed_artifact(tmp_path):
    artifact = {"feature_contract": {
        "version": FEATURE_POLICY_VERSION,
        "feature_names": ["Form"],
        "imputation_values": {"Form": 1.25},
    }}
    artifact_path = tmp_path / "preprocessed_data.pkl"
    with artifact_path.open("wb") as stream:
        pickle.dump(artifact, stream)

    contract = FeatureContract.load(artifact_path)
    assert contract.feature_names == ("Form",)
    assert contract.imputation_values["Form"] == pytest.approx(1.25)


def test_upcoming_matrix_uses_explicit_latest_and_imputed_values():
    history = pd.DataFrame([
        {"MatchDate": "2025-01-01", "HomeTeam": "A", "AwayTeam": "B", "HomeMomentum_L3": 1, "AwayMomentum_L3": 2},
        {"MatchDate": "2025-02-01", "HomeTeam": "A", "AwayTeam": "B", "HomeMomentum_L3": 5, "AwayMomentum_L3": 4},
    ])
    upcoming = pd.DataFrame([
        {"HomeTeam": "A", "AwayTeam": "B", "MatchDate": "2025-03-01", "MarketMargin": 0.08},
        {"HomeTeam": "New", "AwayTeam": "Newer", "MatchDate": "2025-03-01"},
    ])

    matrix = build_upcoming_feature_matrix(history, upcoming, _contract())
    assert np.allclose(matrix, [[5, 4, 0.08], [3, 3, 0.05]])


def test_upcoming_state_is_role_normalized_and_strictly_before_kickoff():
    history = pd.DataFrame([
        {"MatchDate": "2025-01-01", "HomeTeam": "A", "AwayTeam": "B", "HomeMomentum_L3": 1, "AwayMomentum_L3": 2},
        {"MatchDate": "2025-02-01", "HomeTeam": "B", "AwayTeam": "A", "HomeMomentum_L3": 4, "AwayMomentum_L3": 7},
        {"MatchDate": "2025-04-01", "HomeTeam": "A", "AwayTeam": "B", "HomeMomentum_L3": 99, "AwayMomentum_L3": 99},
    ])
    upcoming = pd.DataFrame([{
        "HomeTeam": "A", "AwayTeam": "B", "MatchDate": "2025-03-01"
    }])
    matrix = build_upcoming_feature_matrix(history, upcoming, _contract())
    assert np.allclose(matrix[0], [7, 4, .05])


def test_prediction_frame_normalizes_and_adds_consumer_fields():
    upcoming = pd.DataFrame([{"HomeTeam": "A", "AwayTeam": "B", "HomeGoalsAve": 1.4, "AwayGoalsAve": 1.1}])
    result = build_prediction_frame(upcoming, [[68, 23, 9]])

    assert result.loc[0, "HomeWin_Prob"] == pytest.approx(0.68)
    assert result.loc[0, "PredictedResult"] == "Home Win"
    assert result.loc[0, "Risk_Category"] in {"Low Risk", "Moderate Risk"}
    assert result.loc[0, "Recommendation"] == "Strong Home Lean"
    assert result.loc[0, "ModelLean"] == "A"
    assert result.loc[0, "ModelLeanProbability"] == pytest.approx(0.68)
    assert result.loc[0, "BetRecommendation"] == "No bet"
    assert "Market odds unavailable" in result.loc[0, "BetReason"]
    assert result.loc[0, "Over2_5Prob"] + result.loc[0, "Under2_5Prob"] == pytest.approx(1.0)
    assert result.loc[0, "ExpectedTotalGoals"] == pytest.approx(2.5)
    assert result.loc[0, "PredictionGeneratedAt"].endswith("+00:00")


def test_prediction_frame_uses_xg_features_for_goal_markets():
    upcoming = pd.DataFrame([{
        "HomeTeam": "A", "AwayTeam": "B", "HomexG_Avg_L5": 1.6, "AwayxG_Avg_L5": 0.9,
    }])
    result = build_prediction_frame(upcoming, [[50, 30, 20]])

    assert result.loc[0, "ExpectedTotalGoals"] == pytest.approx(2.5)
    assert result.loc[0, "Over2_5Prob"] + result.loc[0, "Under2_5Prob"] == pytest.approx(1.0)


def test_prediction_frame_rejects_invalid_shape():
    with pytest.raises(ValueError):
        build_prediction_frame(pd.DataFrame([{"HomeTeam": "A", "AwayTeam": "B"}]), [[0.5, 0.5]])
