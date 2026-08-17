"""Consumer-facing feature and prediction-cache contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .features import FEATURE_POLICY_VERSION
from .goal_markets import calculate_goal_markets
from .risk import calculate_prediction_risk, get_prediction_guidance, get_risk_category


@dataclass(frozen=True)
class FeatureContract:
    """Exact feature order and preprocessing values used by a trained model."""

    version: int
    feature_names: tuple[str, ...]
    imputation_values: Mapping[str, float]
    state_sources: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version != FEATURE_POLICY_VERSION:
            raise ValueError(
                f"Feature contract version {self.version} is incompatible with required "
                f"version {FEATURE_POLICY_VERSION}"
            )
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Feature names must be non-empty and unique")
        missing = [name for name in self.feature_names if name not in self.imputation_values]
        if missing:
            raise ValueError(f"Feature contract lacks imputation values for: {', '.join(missing)}")
        unknown = set(self.state_sources).difference(self.feature_names)
        if unknown:
            raise ValueError(f"State sources reference unknown features: {sorted(unknown)}")
        for feature, source in self.state_sources.items():
            required = {"fixture_role", "home_history_column", "away_history_column"}
            if required.difference(source):
                raise ValueError(f"State source for {feature} is incomplete")
            if source["fixture_role"] not in {"home", "away"}:
                raise ValueError(f"State source for {feature} has invalid fixture_role")
            if not source["home_history_column"] or not source["away_history_column"]:
                raise ValueError(f"State source for {feature} has an empty history column")

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, object]) -> "FeatureContract":
        payload = artifact.get("feature_contract")
        if not isinstance(payload, Mapping):
            raise ValueError("Precomputed artifact has no feature contract; regenerate it")
        version = int(payload.get("version", -1))
        names = tuple(str(name) for name in payload.get("feature_names", ()))
        values = payload.get("imputation_values", {})
        if not names or not isinstance(values, Mapping):
            raise ValueError("Feature contract is incomplete")
        missing = [name for name in names if name not in values]
        if missing:
            raise ValueError(f"Feature contract lacks imputation values for: {', '.join(missing)}")
        raw_sources = payload.get("state_sources", {})
        if not isinstance(raw_sources, Mapping):
            raise ValueError("Feature state_sources must be a mapping")
        invalid_sources = [
            str(feature)
            for feature, source in raw_sources.items()
            if not isinstance(source, Mapping)
        ]
        if invalid_sources:
            raise ValueError(
                f"Feature state_sources contain invalid mappings: {sorted(invalid_sources)}"
            )
        state_sources = {
            str(feature): {str(key): str(value) for key, value in source.items()}
            for feature, source in raw_sources.items()
        }
        return cls(
            version, names, {name: float(values[name]) for name in names}, state_sources
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureContract":
        with Path(path).open("rb") as stream:
            artifact = pickle.load(stream)
        if not isinstance(artifact, Mapping):
            raise ValueError("Precomputed artifact must contain a mapping")
        return cls.from_artifact(artifact)


def build_upcoming_feature_matrix(
    historical: pd.DataFrame,
    upcoming: pd.DataFrame,
    contract: FeatureContract,
) -> np.ndarray:
    """Build live features in the exact order used during model training.

    Explicit values already present on an upcoming fixture take priority. Team-
    specific fields use explicit perspective mappings in the artifact contract.
    State lookup is strictly before each fixture kickoff; unmapped or unavailable
    values use training-period imputations. No name-prefix inference, padding, or
    truncation is permitted.
    """
    required = {"HomeTeam", "AwayTeam"}
    if not required.issubset(historical.columns) or not required.issubset(upcoming.columns):
        raise ValueError("Historical and upcoming data require HomeTeam and AwayTeam")
    if upcoming.empty:
        return np.empty((0, len(contract.feature_names)), dtype=np.float32)

    ordered_history = historical.copy()
    date_column = next(
        (column for column in ("kickoff_utc", "MatchDate") if column in ordered_history),
        None,
    )
    if contract.state_sources and date_column is None:
        raise ValueError("Explicit state lookup requires a historical kickoff/date column")
    required_history_columns = {
        source[column]
        for source in contract.state_sources.values()
        for column in ("home_history_column", "away_history_column")
    }
    missing_history_columns = required_history_columns.difference(ordered_history.columns)
    if missing_history_columns:
        raise ValueError(
            f"Historical data lacks contracted state columns: {sorted(missing_history_columns)}"
        )
    if date_column is not None:
        ordered_history = ordered_history.assign(
            _contract_date=pd.to_datetime(
                ordered_history[date_column], utc=True, errors="coerce"
            )
        ).sort_values("_contract_date", kind="stable")

    output: list[list[float]] = []
    for _, match in upcoming.iterrows():
        home, away = str(match["HomeTeam"]), str(match["AwayTeam"])
        cutoff_value = next(
            (match.get(column) for column in ("kickoff_utc", "MatchDate") if pd.notna(match.get(column))),
            None,
        )
        cutoff = pd.to_datetime(cutoff_value, utc=True, errors="coerce")
        if contract.state_sources and pd.isna(cutoff):
            raise ValueError("Explicit state lookup requires every upcoming fixture kickoff")
        available_history = (
            ordered_history.loc[ordered_history._contract_date < cutoff]
            if contract.state_sources else ordered_history
        )
        values: list[float] = []
        for feature in contract.feature_names:
            value = match.get(feature)
            source = contract.state_sources.get(feature)
            if pd.isna(value) and source:
                target_team = home if source["fixture_role"] == "home" else away
                home_rows = available_history.loc[
                    available_history.HomeTeam.astype(str) == target_team,
                    ["_contract_date", source["home_history_column"]],
                ].rename(columns={source["home_history_column"]: "_state_value"})
                away_rows = available_history.loc[
                    available_history.AwayTeam.astype(str) == target_team,
                    ["_contract_date", source["away_history_column"]],
                ].rename(columns={source["away_history_column"]: "_state_value"})
                state = pd.concat([home_rows, away_rows], ignore_index=True)
                state = state.dropna(subset=["_contract_date", "_state_value"]).sort_values(
                    "_contract_date", kind="stable"
                )
                if not state.empty:
                    value = state.iloc[-1]._state_value
            if pd.isna(value):
                value = contract.imputation_values[feature]
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature {feature!r} is not numeric for {home} vs {away}") from exc
            if not np.isfinite(numeric_value):
                numeric_value = contract.imputation_values[feature]
            values.append(numeric_value)
        output.append(values)

    matrix = np.asarray(output, dtype=np.float32)
    expected_shape = (len(upcoming), len(contract.feature_names))
    if matrix.shape != expected_shape:
        raise ValueError(f"Feature matrix has shape {matrix.shape}; expected {expected_shape}")
    return matrix


def build_prediction_frame(
    upcoming: pd.DataFrame,
    probabilities: Sequence[Sequence[float]],
) -> pd.DataFrame:
    """Attach normalized 1X2 probabilities, risk, and guidance to fixtures."""
    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.shape != (len(upcoming), 3):
        raise ValueError(
            f"Probability matrix has shape {probability_array.shape}; expected {(len(upcoming), 3)}"
        )
    if not np.isfinite(probability_array).all() or (probability_array < 0).any():
        raise ValueError("Probabilities must be finite and non-negative")
    totals = probability_array.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Every prediction row needs positive probability mass")
    probability_array = probability_array / totals

    result = upcoming.copy()
    result[["HomeWin_Prob", "Draw_Prob", "AwayWin_Prob"]] = probability_array
    labels = np.asarray(["Home Win", "Draw", "Away Win"])
    best_indices = np.argmax(probability_array, axis=1)
    result["PredictedResult"] = labels[best_indices]
    result["ModelLean"] = [
        str(result.iloc[position]["HomeTeam"])
        if best_index == 0
        else str(result.iloc[position]["AwayTeam"])
        if best_index == 2
        else "Draw"
        for position, best_index in enumerate(best_indices)
    ]
    result["ModelLeanProbability"] = probability_array.max(axis=1)

    risks, confidences, categories, guidance = [], [], [], []
    for row in probability_array:
        risk, confidence = calculate_prediction_risk(row)
        category, _ = get_risk_category(risk)
        recommendation, _ = get_prediction_guidance(row, risk)
        risks.append(risk)
        confidences.append(confidence)
        categories.append(category)
        guidance.append(recommendation)
    result["Risk_Score"] = risks
    result["Confidence_Score"] = confidences
    result["Risk_Category"] = categories
    result["Recommendation"] = guidance
    result["BetRecommendation"] = "No bet"
    result["BetReason"] = "Market odds unavailable; betting value cannot be established."
    add_goal_market_predictions(result)
    result["PredictionGeneratedAt"] = pd.Timestamp.now(tz="UTC").isoformat()
    return result


def production_probabilities(
    historical: pd.DataFrame,
    upcoming: pd.DataFrame,
    contract: FeatureContract,
    *,
    production_candidate: str,
    models_dir: str | Path = "models",
) -> np.ndarray:
    """Return 1X2 probabilities from the audit-selected production candidate.

    ``no_odds`` loads the fitted ``ensemble_model.pkl`` and builds the
    point-in-time feature matrix; ``poisson`` uses the walk-forward expected
    goals model over completed history only.  Anything else is refused so the
    audit gate and the shipped cache can never disagree about the model.
    """
    if production_candidate == "no_odds":
        model_path = Path(models_dir) / "ensemble_model.pkl"
        with model_path.open("rb") as stream:
            model = pickle.load(stream)
        matrix = build_upcoming_feature_matrix(historical, upcoming, contract)
        probabilities = model.predict_proba(matrix)
        if probabilities.shape != (len(upcoming), 3):
            raise ValueError(
                f"No-odds model returned shape {probabilities.shape}; "
                f"expected {(len(upcoming), 3)}"
            )
        return probabilities
    if production_candidate == "poisson":
        from models.poisson_evaluation import predict_upcoming_outcomes

        outcome_probabilities, expected_goals = predict_upcoming_outcomes(historical, upcoming)
        probabilities = np.asarray(outcome_probabilities)
        if probabilities.shape != (len(upcoming), 3):
            raise ValueError(
                f"Poisson model returned shape {probabilities.shape}; "
                f"expected {(len(upcoming), 3)}"
            )
        upcoming["ExpectedHomeGoals"] = [pair[0] for pair in expected_goals]
        upcoming["ExpectedAwayGoals"] = [pair[1] for pair in expected_goals]
        return probabilities
    raise ValueError(
        f"Unknown production candidate {production_candidate!r}; "
        "expected 'no_odds' or 'poisson'"
    )


def add_goal_market_predictions(result: pd.DataFrame) -> None:
    """Add common goal-market probabilities when expected goals are available.

    The consumer cache may contain either model-derived expected goals or the
    historical feature averages used by the model.  If neither pair exists,
    the 1X2 prediction contract remains unchanged and the UI simply omits the
    optional goal-market columns.
    """
    goal_columns = (
        ("Expected_Home_Goals", "Expected_Away_Goals"),
        ("Model_Expected_Home_Goals", "Model_Expected_Away_Goals"),
        ("ExpectedHomeGoals", "ExpectedAwayGoals"),
        ("HomeGoalsAve", "AwayGoalsAve"),
        ("HomexG_Avg_L5", "AwayxG_Avg_L5"),
    )
    source = next(
        ((home, away) for home, away in goal_columns if home in result and away in result),
        None,
    )
    if source is None:
        return

    home_goals = pd.to_numeric(result[source[0]], errors="coerce")
    away_goals = pd.to_numeric(result[source[1]], errors="coerce")
    market_rows = []
    for home, away in zip(home_goals, away_goals):
        if pd.isna(home) or pd.isna(away) or home < 0 or away < 0:
            market_rows.append({})
            continue
        market_rows.append(calculate_goal_markets(float(home), float(away), lines=(2.5,)).as_dict())

    market_frame = pd.DataFrame(market_rows, index=result.index)
    for column in (
        "ExpectedTotalGoals", "Over2_5Prob", "Under2_5Prob", "BTTSProb",
        "MostLikelyScore",
    ):
        if column in market_frame:
            result[column] = market_frame[column]
