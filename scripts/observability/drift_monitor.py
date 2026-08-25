"""Drift monitor — PSI on features, alert if any window exceeds threshold.

Run: python scripts/observability/drift_monitor.py
"""

import json
import os
from pathlib import Path

import pandas as pd

from pitch_oracle_core.evaluation.drift_monitor import (
    drift_report,
    monitor_feature_drift,
)


def main():
    data_dir = Path(os.getenv("PITCH_ORACLE_DATA_DIR", "data_files"))
    features_path = data_dir / "combined_historical_data_with_calculations_new.csv"
    if not features_path.exists():
        print(f"No features at {features_path}")
        return

    df = pd.read_csv(features_path, sep="\t")
    # Reference: first 60% of data, Current: last 40%
    split = int(len(df) * 0.6)
    reference = df.iloc[:split]
    current = df.iloc[split:]

    numeric_cols = [c for c in df.columns if df[c].dtype in ("float64", "int64") and c not in ("Season",)]
    findings = monitor_feature_drift(reference, current, features=numeric_cols[:20])
    report = drift_report(findings)

    print("Drift Monitor Report:")
    print(report.to_string(index=False))

    action_required = [f for f in findings if f.severity == "action_required"]
    if action_required:
        print(f"\n⚠ {len(action_required)} features require action!")
    else:
        print("\n✓ All features stable")

    output = data_dir / "observability"
    output.mkdir(parents=True, exist_ok=True)
    severity = {
        "status": "critical" if action_required else "ok",
        "features_checked": len(findings),
        "action_required": len(action_required),
        "watch": len([f for f in findings if f.severity == "watch"]),
    }
    (output / "severity.json").write_text(json.dumps(severity, indent=2))
    report.to_json(output / "input_drift.json", orient="records", indent=2)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
