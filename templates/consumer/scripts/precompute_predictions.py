"""Build the strict upcoming-prediction artifact from the shared feature contract."""

from pathlib import Path
import pickle

import pandas as pd

from pitch_oracle_core import (
    FeatureContract,
    build_prediction_frame,
    build_upcoming_feature_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def generate() -> Path:
    historical = pd.read_csv(
        ROOT / "data_files" / "combined_historical_data_with_calculations_new.csv",
        sep="\t",
    )
    upcoming = pd.read_csv(ROOT / "data_files" / "upcoming_fixtures.csv")
    contract = FeatureContract.load(ROOT / "precomputed" / "preprocessed_data.pkl")
    with (ROOT / "models" / "ensemble_model.pkl").open("rb") as stream:
        model = pickle.load(stream)
    matrix = build_upcoming_feature_matrix(historical, upcoming, contract)
    output = ROOT / "data_files" / "upcoming_predictions.csv"
    build_prediction_frame(upcoming, model.predict_proba(matrix)).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    print(f"Wrote {generate()}")
