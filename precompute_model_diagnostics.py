"""Precompute reusable model-diagnostic artifacts for the Streamlit app."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from models.feature_analysis import analyze_feature_importance_shap


DROP_COLS = {
    "FullTimeResult", "FullTimeHomeGoals", "FullTimeAwayGoals",
    "HalfTimeResult", "HalfTimeHomeGoals", "HalfTimeAwayGoals",
    "HomeWin", "AwayWin", "Draw", "WinningTeam", "HomePoints", "AwayPoints",
    "HomeTeamCumulativePoints", "AwayTeamCumulativePoints", "MatchDate",
    "KickoffTime", "Season", "Round", "Venue", "Referee", "HomeTeam",
    "AwayTeam", "Division", "target",
}


def _feature_names(frame: pd.DataFrame, saved_names: list[str]) -> list[str]:
    if saved_names and not all(str(name).startswith("feature_") for name in saved_names):
        return saved_names
    usable = frame.drop(columns=[c for c in DROP_COLS if c in frame], errors="ignore")
    names = list(usable.select_dtypes(include=[np.number]).columns)
    names += list(usable.select_dtypes(include=["object"]).columns)
    return names if len(names) == len(saved_names) else saved_names


def generate() -> Path:
    root = Path.cwd()
    data_dir = Path(os.getenv("PITCH_ORACLE_DATA_DIR", root / "data_files"))
    models_dir = Path(os.getenv("PITCH_ORACLE_MODELS_DIR", root / "models"))
    output_dir = Path(os.getenv("PITCH_ORACLE_DIAGNOSTICS_DIR", root / "precomputed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = data_dir / "combined_historical_data_with_calculations_new.csv"
    precomputed_path = root / "precomputed" / "preprocessed_data.pkl"
    model_path = next(
        (models_dir / name for name in ("xgb_baseline.pkl", "optimized_xgb.pkl", "ensemble_model.pkl")
         if (models_dir / name).exists()),
        None,
    )
    if not source_path.exists() or not precomputed_path.exists() or model_path is None:
        raise FileNotFoundError("Diagnostics require source data, precomputed data, and a trained model")

    with precomputed_path.open("rb") as stream:
        cached = pickle.load(stream)
    with model_path.open("rb") as stream:
        model = pickle.load(stream)

    frame = pd.read_csv(source_path, sep="\t")
    names = _feature_names(frame, list(cached["feature_names"]))
    X_test = np.asarray(cached["X_test"])
    y_test = np.asarray(cached["y_test"])

    permutation = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1,
    )
    permutation_df = pd.DataFrame({
        "Feature": names,
        "Importance": permutation.importances_mean,
        "Std": permutation.importances_std,
    }).sort_values("Importance", ascending=False)
    permutation_df.to_csv(output_dir / "feature_importance.csv", index=False)

    bar_fig, class_figs, _, shap_df = analyze_feature_importance_shap(
        model, X_test, names, max_display=20,
    )
    shap_df.to_csv(output_dir / "shap_importance.csv", index=False)
    bar_fig.savefig(output_dir / "shap_overall.png", dpi=150, bbox_inches="tight")
    class_names = ("home_win", "draw", "away_win")
    for name, figure in zip(class_names, class_figs):
        figure.savefig(output_dir / f"shap_{name}.png", dpi=150, bbox_inches="tight")

    (output_dir / "model_diagnostics.json").write_text(json.dumps({
        "model": model_path.name,
        "features": len(names),
        "test_samples": len(X_test),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }, indent=2), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    print(f"Wrote model diagnostics to {generate()}")
