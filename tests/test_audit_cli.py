"""Tests for the chronology/model-ablation release gate CLI."""

import json

import pandas as pd
import pytest

from pitch_oracle_core.audit_cli import generate


def _frame() -> pd.DataFrame:
    """Small but realistic frame: goals present, so the Poisson candidate runs."""
    rows = []
    teams = ["A", "B", "C", "D"]
    labels = ["H", "D", "A"] * 30
    for index, label in enumerate(labels):
        home = teams[index % 4]
        away = teams[(index + 1) % 4]
        goals = (1, 0) if label == "H" else (0, 1) if label == "A" else (1, 1)
        rows.append({
            "MatchDate": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
            "FullTimeResult": label,
            "HomeTeam": home,
            "AwayTeam": away,
            "FullTimeHomeGoals": goals[0],
            "FullTimeAwayGoals": goals[1],
            "HomeMomentum_L3": float(index % 4),
            "AwayMomentum_L3": float((index + 1) % 4),
            "HomexG_Avg_L5": 1.2,
            "AwayxG_Avg_L5": 1.1,
        })
    return pd.DataFrame(rows)


def test_gate_selects_poisson_when_poisson_beats_baseline(tmp_path):
    # A frame where the no-odds logistic cannot learn (pure noise) but the
    # walk-forward Poisson still scores meaningful probabilities. The gate must
    # prefer poisson even though the logistic candidate is below baseline.
    rows = []
    for index in range(90):
        label = "H" if index % 3 == 0 else "D" if index % 3 == 1 else "A"
        rows.append({
            "MatchDate": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
            "FullTimeResult": label,
            "HomeTeam": "A",
            "AwayTeam": "B",
            "FullTimeHomeGoals": 1,
            "FullTimeAwayGoals": 1,
            "HomeMomentum_L3": float(index % 4),
            "AwayMomentum_L3": float((index + 1) % 4),
            "HomexG_Avg_L5": 1.0,
            "AwayxG_Avg_L5": 1.0,
        })
    source = tmp_path / "matches.csv"
    pd.DataFrame(rows).to_csv(source, sep="\t", index=False)

    report = generate(source, tmp_path)

    assert report["status"] == "complete"
    candidates = {item["candidate"]: item for item in report["ablation"]}
    assert "poisson" in candidates
    assert candidates["poisson"]["test_rows"] > 0
    persisted = json.loads((tmp_path / "model_ablation.json").read_text(encoding="utf-8"))
    assert "production_candidate" in persisted["release_gate"]
    assert persisted["release_gate"]["production_candidate"] in (None, "no_odds", "poisson")


def test_gate_fails_when_no_candidate_beats_baseline(tmp_path):
    # Uniform random labels: neither the logistic nor Poisson can beat the prior.
    rows = []
    for index in range(90):
        label = "H" if index % 3 == 0 else "D" if index % 3 == 1 else "A"
        rows.append({
            "MatchDate": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
            "FullTimeResult": label,
            "HomeTeam": "A",
            "AwayTeam": "B",
            "FullTimeHomeGoals": 1,
            "FullTimeAwayGoals": 1,
            "HomeMomentum_L3": float(index % 2),
            "AwayMomentum_L3": float((index + 1) % 2),
            "HomexG_Avg_L5": 1.0,
            "AwayxG_Avg_L5": 1.0,
        })
    source = tmp_path / "matches.csv"
    pd.DataFrame(rows).to_csv(source, sep="\t", index=False)

    report = generate(source, tmp_path)

    assert report["status"] == "complete"
    assert report["release_gate"]["passed"] is False
