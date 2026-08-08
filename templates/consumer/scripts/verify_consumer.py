"""Strict league-neutral artifact and chronological model-quality gate."""

from __future__ import annotations

import math
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import LEAGUE_CONFIG
from pitch_oracle_core import FeatureContract, __version__
from pitch_oracle_core.cache import validate_cache


def main() -> None:
    validate_cache(ROOT, expected_league=LEAGUE_CONFIG.key)
    contract = FeatureContract.load(ROOT / "precomputed" / "preprocessed_data.pkl")
    with (ROOT / "models" / "ensemble_model.pkl").open("rb") as stream:
        ensemble = pickle.load(stream)
    width = getattr(ensemble, "n_features_in_", None)
    if width is not None and width != len(contract.feature_names):
        raise SystemExit(
            f"Ensemble width {width} does not match contract width "
            f"{len(contract.feature_names)}"
        )

    with (ROOT / "models" / "model_performance.pkl").open("rb") as stream:
        performance = pickle.load(stream)
    required = {"class_prior_baseline", "xgb_baseline", "ensemble", "optimized_xgb", "poisson"}
    missing = required.difference(performance)
    if missing:
        raise SystemExit(f"Missing model metrics: {sorted(missing)}")
    for name in ("xgb_baseline", "ensemble", "optimized_xgb"):
        accuracy = float(performance[name]["accuracy"])
        log_loss = float(performance[name]["log_loss"])
        if not (0.0 <= accuracy <= 1.0 and math.isfinite(log_loss) and log_loss < 2.0):
            raise SystemExit(f"Implausible chronological metrics for {name}: {performance[name]}")
    production = performance["ensemble"]
    baseline = performance["class_prior_baseline"]
    if (
        float(production["log_loss"]) >= float(baseline["log_loss"])
        or float(production["brier_score"]) >= float(baseline["brier_score"])
    ):
        raise SystemExit(
            "Production no-odds model does not beat the class-prior baseline "
            "on log loss and Brier score"
        )
    poisson_accuracy = float(performance["poisson"]["outcome_acc"])
    if not 0.0 <= poisson_accuracy <= 1.0:
        raise SystemExit(f"Invalid Poisson outcome accuracy: {poisson_accuracy}")

    print(f"{LEAGUE_CONFIG.display_name} artifacts verified with core {__version__}")
    print(f"Feature contract width: {len(contract.feature_names)}")


if __name__ == "__main__":
    main()
