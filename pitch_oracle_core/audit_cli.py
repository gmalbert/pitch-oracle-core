"""CLI for the chronology, feature-ablation, and model release gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from models.no_odds_predictor import create_no_odds_classifier
from .model_audit import data_quality_summary, evaluate_feature_ablation, feature_inventory


def _estimator():
    return create_no_odds_classifier()


def generate(source: Path, output_dir: Path, *, as_of: str | None = None) -> dict:
    frame = pd.read_csv(source, sep="\t")
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_inventory(frame).to_csv(output_dir / "feature_inventory.csv", index=False)
    quality = data_quality_summary(frame, as_of=as_of)
    report: dict = {"source": str(source), "data_quality": quality}
    if quality["invalid_dates"] or quality["completed_future_rows"]:
        report.update({
            "status": "blocked_invalid_chronology",
            "action": (
                "Regenerate history from raw source dates with an explicit date format; "
                "do not use this dataset for chronological model selection."
            ),
            "ablation": [],
        })
    else:
        results = evaluate_feature_ablation(frame, _estimator)
        by_name = {result.candidate: result for result in results}
        baseline = by_name["class_prior_baseline"].metrics
        no_odds = by_name["no_odds"].metrics
        passed = (
            no_odds["log_loss"] < baseline["log_loss"]
            and no_odds["brier_score"] < baseline["brier_score"]
        )
        report.update({
            "status": "complete",
            "release_gate": {
                "passed": passed,
                "criteria": "no_odds beats rolling class-prior baseline on log loss and Brier score",
            },
            "ablation": [asdict(result) for result in results],
        })
    (output_dir / "model_ablation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/model-audit"))
    parser.add_argument("--as-of")
    args = parser.parse_args()
    report = generate(args.source, args.output_dir, as_of=args.as_of)
    print(json.dumps(report, indent=2))
    gate = report.get("release_gate", {})
    if report["status"] != "complete" or gate.get("passed") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
