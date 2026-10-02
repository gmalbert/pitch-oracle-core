"""Paired walk-forward PitchAPI ablations and conservative per-league promotion."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitch_oracle_core.features import chronological_split_indices, no_odds_feature_columns
from pitch_oracle_core.features.advanced_team import assert_point_in_time_features
from pitch_oracle_core.features.families import FAMILIES, FeatureFamilyConfig, feature_family
from pitch_oracle_core.model_audit import TemperatureScaledClassifier, fit_temperature, probability_metrics, rolling_origin_splits
from pitch_oracle_core.pitchapi.cache import atomic_json
from pitch_oracle_core.pitchapi.storage import read_frame, write_frame

ABLATIONS = {f"A{index}": (family,) for index, family in enumerate(FAMILIES, start=1)}
ABLATIONS = {"A0": (), **ABLATIONS, "A9": FAMILIES}


@dataclass(frozen=True)
class PromotionPolicy:
    minimum_paired_fixtures: int = 200
    minimum_week_blocks: int = 10
    minimum_coverage: float = .80
    minimum_train_rows: int = 150
    maximum_calibration_degradation: float = .01
    maximum_subgroup_log_loss_degradation: float = .03
    minimum_log_loss_improvement: float = .005
    minimum_brier_improvement: float = .01
    bootstrap_repetitions: int = 2000
    confidence: float = .99375


def per_fixture_scores(y, probabilities) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(y), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Invalid paired probabilities or outcomes")
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)), np.sum((p - np.eye(3)[y]) ** 2, axis=1)


def paired_block_bootstrap(baseline: np.ndarray, candidate: np.ndarray, blocks, *, repetitions: int = 2000, confidence: float = .99375, seed: int = 42) -> dict:
    """Resample whole calendar weeks, preserving dependence and fixture pairing."""
    baseline, candidate = np.asarray(baseline, float), np.asarray(candidate, float)
    blocks = np.asarray(blocks)
    if baseline.shape != candidate.shape or baseline.ndim != 1 or len(blocks) != len(baseline) or not len(baseline) or not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("Bootstrap requires equal finite paired score vectors")
    if repetitions < 100 or not .5 < confidence < 1:
        raise ValueError("Invalid bootstrap repetitions or confidence")
    groups = [np.flatnonzero(blocks == block) for block in np.unique(blocks)]
    random = np.random.default_rng(seed)
    values = np.empty(repetitions)
    for repeat in range(repetitions):
        indices = np.concatenate([groups[index] for index in random.integers(0, len(groups), len(groups))])
        denominator = baseline[indices].mean()
        values[repeat] = (denominator - candidate[indices].mean()) / denominator if denominator > 0 else 0
    alpha = (1 - confidence) / 2
    return {"relative_improvement": float((baseline.mean() - candidate.mean()) / baseline.mean()) if baseline.mean() > 0 else 0, "lower": float(np.quantile(values, alpha)), "upper": float(np.quantile(values, 1 - alpha)), "confidence": confidence, "week_blocks": len(groups), "paired_fixtures": len(baseline), "seed": seed, "repetitions": repetitions}


def _availability(frame: pd.DataFrame, families: tuple[str, ...], columns: list[str]) -> pd.Series:
    available = pd.Series(True, index=frame.index)
    for family in families:
        names = [column for column in columns if feature_family(column) == family and "coverage" not in column and "is_confirmed" not in column]
        if not names:
            return available & False
        # At least one real measure for each family; imputation is still fitted
        # on training rows only. Coverage flags alone do not count as data.
        available &= frame[names].notna().any(axis=1)
        if family == "confirmed_lineup":
            available &= frame.get("home_lineup_status", pd.Series("", index=frame.index)).eq("confirmed") | frame.get("away_lineup_status", pd.Series("", index=frame.index)).eq("confirmed")
    return available


def _fold_probabilities(frame: pd.DataFrame, columns: list[str], train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, dict]:
    from models.no_odds_predictor import create_no_odds_classifier

    fit_relative, calibration_relative = chronological_split_indices(frame.iloc[train].MatchDate, test_size=.2)
    fit, calibration = train[fit_relative], train[calibration_relative]
    y = frame.FullTimeResult.map({"H": 0, "D": 1, "A": 2}).astype(int)
    if set(y.iloc[fit]) != {0, 1, 2}:
        raise ValueError("Fold training period lacks all outcome classes")
    x = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    means = x.iloc[fit].mean().fillna(0)
    x = x.fillna(means).to_numpy()
    estimator = create_no_odds_classifier().fit(x[fit], y.iloc[fit].to_numpy())
    temperature = fit_temperature(estimator.predict_proba(x[calibration]), y.iloc[calibration].to_numpy())
    model = TemperatureScaledClassifier(estimator, temperature)
    return model.predict_proba(x[test]), {"fit_rows": len(fit), "calibration_rows": len(calibration), "test_rows": len(test), "fit_through": str(frame.iloc[fit].MatchDate.max()), "calibration_through": str(frame.iloc[calibration].MatchDate.max()), "test_from": str(frame.iloc[test].MatchDate.min()), "temperature": temperature}


def _comparison(paired: pd.DataFrame, *, coverage: float, policy: PromotionPolicy, strict: bool, mapping_gate: bool) -> dict:
    y = paired.target.to_numpy()
    baseline = paired[["baseline_home", "baseline_draw", "baseline_away"]].to_numpy()
    candidate = paired[["p_home", "p_draw", "p_away"]].to_numpy()
    baseline_loss, baseline_brier = per_fixture_scores(y, baseline)
    loss, brier = per_fixture_scores(y, candidate)
    weeks = pd.to_datetime(paired.MatchDate, utc=True).dt.strftime("%G-W%V")
    loss_ci = paired_block_bootstrap(baseline_loss, loss, weeks, repetitions=policy.bootstrap_repetitions, confidence=policy.confidence)
    brier_ci = paired_block_bootstrap(baseline_brier, brier, weeks, repetitions=policy.bootstrap_repetitions, confidence=policy.confidence)
    base_metrics, metrics = probability_metrics(y, baseline), probability_metrics(y, candidate)
    target_cumulative = np.cumsum(np.eye(3)[y], axis=1)[:, :2]
    base_metrics["rps"] = float(np.mean(np.sum((np.cumsum(baseline, axis=1)[:, :2] - target_cumulative) ** 2, axis=1) / 2))
    metrics["rps"] = float(np.mean(np.sum((np.cumsum(candidate, axis=1)[:, :2] - target_cumulative) ** 2, axis=1) / 2))
    reliability = {}
    for label, p in (("baseline", baseline), ("candidate", candidate)):
        reliability[label] = {}
        for outcome, name in enumerate(("home", "draw", "away")):
            rows = []
            for bucket in range(10):
                mask = np.minimum((p[:, outcome] * 10).astype(int), 9) == bucket
                if mask.any():
                    rows.append({"forecast_mean": float(p[mask, outcome].mean()), "observed_rate": float((y[mask] == outcome).mean()), "fixtures": int(mask.sum())})
            reliability[label][name] = rows
    subgroups = []
    paired = paired.assign(_baseline_loss=baseline_loss, _candidate_loss=loss)
    for dimension, values in (("result_class", paired.target.astype(str)), ("calendar_month", pd.to_datetime(paired.MatchDate).dt.strftime("%Y-%m")), ("home_team", paired.HomeTeam), ("away_team", paired.AwayTeam)):
        for group, rows in paired.assign(_group=values).groupby("_group", sort=True):
            base, new = rows._baseline_loss.mean(), rows._candidate_loss.mean()
            subgroups.append({"dimension": dimension, "group": str(group), "fixtures": len(rows), "relative_log_loss_improvement": float((base - new) / base) if base else 0, "gate_evaluated": len(rows) >= 30})
    criteria = {
        "strict_observation_replay": strict,
        "fixture_mapping": mapping_gate,
        "paired_sample": len(paired) >= policy.minimum_paired_fixtures,
        "independent_week_blocks": loss_ci["week_blocks"] >= policy.minimum_week_blocks,
        "coverage": coverage >= policy.minimum_coverage,
        "improvement_with_uncertainty": (loss_ci["relative_improvement"] >= policy.minimum_log_loss_improvement and loss_ci["lower"] > 0) or (brier_ci["relative_improvement"] >= policy.minimum_brier_improvement and brier_ci["lower"] > 0),
        "proper_score_stability": loss_ci["relative_improvement"] >= -.005 and brier_ci["relative_improvement"] >= -.01,
        "calibration": metrics["calibration_error"] <= base_metrics["calibration_error"] + policy.maximum_calibration_degradation,
        "subgroups": all(row["relative_log_loss_improvement"] >= -policy.maximum_subgroup_log_loss_degradation for row in subgroups if row["gate_evaluated"]),
    }
    return {"baseline": base_metrics, "candidate": metrics, "reliability": reliability, "log_loss_uncertainty": loss_ci, "brier_uncertainty": brier_ci, "subgroups": subgroups, "coverage": coverage, "criteria": criteria, "promotion_passed": all(criteria.values())}


def evaluate_pitchapi_families(frame: pd.DataFrame, *, league_key: str, n_splits: int = 4, policy: PromotionPolicy | None = None, strict: bool = True, mapping_gate: bool = False, selected_families: tuple[str, ...] = ()) -> tuple[dict, pd.DataFrame]:
    policy = policy or PromotionPolicy()
    required = {"fixture_id", "kickoff_utc", "MatchDate", "HomeTeam", "AwayTeam", "FullTimeResult", "as_of", "feature_observed_at"}
    if required.difference(frame):
        raise ValueError(f"Ablation data misses {sorted(required.difference(frame))}")
    if frame.duplicated(["fixture_id", "as_of"]).any() or not frame.FullTimeResult.isin(["H", "D", "A"]).all():
        raise ValueError("Ablation requires unique fixture/cutoff rows with completed outcomes")
    if "league_key" in frame and not frame.league_key.eq(league_key).all():
        raise ValueError("Promotion data belongs to a different league")
    if strict:
        assert_point_in_time_features(frame)
    if set(selected_families).difference(FAMILIES):
        raise ValueError("Unknown combined family selection")
    frame = frame.sort_values(["MatchDate", "fixture_id"], kind="stable").reset_index(drop=True)
    frame["target"] = frame.FullTimeResult.map({"H": 0, "D": 1, "A": 2})
    stages = frame.get("forecast_stage", pd.Series("pre_kickoff_features", index=frame.index)).astype(str)
    if stages.nunique() != 1:
        raise ValueError("Evaluate forecast stages separately on paired fixture/cutoff rows")
    folds = rolling_origin_splits(frame.MatchDate, n_splits=n_splits, minimum_train_fraction=.5)
    report = {"schema_version": 1, "league_key": league_key, "forecast_stage": stages.iloc[0], "evaluation_mode": "strict_observation_replay" if strict else "retrospective", "policy": asdict(policy), "ablations": {}, "recommended_families": [], "chronology_violations": 0 if strict else None}
    baseline_columns = frame[no_odds_feature_columns(frame)].select_dtypes(include=[np.number]).columns.tolist()
    if not baseline_columns:
        raise ValueError("Ablation requires baseline numeric features")
    all_predictions = []
    ablations = {**ABLATIONS, "A9": selected_families}
    for name, families in ablations.items():
        columns = frame[no_odds_feature_columns(frame, enabled_families=families)].select_dtypes(include=[np.number]).columns.tolist()
        if name == "A9" and not families:
            report["ablations"][name] = {"status": "not_selected", "families": []}
            continue
        availability = _availability(frame, families, columns)
        records, fold_records = [], []
        for index, (train, test) in enumerate(folds):
            if len(train) < policy.minimum_train_rows:
                continue
            baseline_p, fold_metadata = _fold_probabilities(frame, baseline_columns, train, test)
            candidate_p = baseline_p if not families else _fold_probabilities(frame, columns, train, test)[0]
            test_rows = frame.iloc[test][["fixture_id", "as_of", "MatchDate", "HomeTeam", "AwayTeam", "target"]].copy()
            test_rows[["baseline_home", "baseline_draw", "baseline_away"]] = baseline_p
            test_rows[["p_home", "p_draw", "p_away"]] = candidate_p
            test_rows["family_available"] = availability.iloc[test].to_numpy()
            test_rows["fold"] = index
            test_rows["ablation"] = name
            records.append(test_rows)
            fold_records.append({"fold": index, **fold_metadata})
        predictions = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        record = {"families": list(families), "feature_count": len(columns), "folds": fold_records, "status": "insufficient_data", "promotion_passed": False}
        if not predictions.empty:
            all_predictions.append(predictions)
            paired = predictions.loc[predictions.family_available]
            record["expected_test_fixtures"] = len(predictions)
            record["paired_fixtures"] = len(paired)
            record["missingness"] = {}
            for dimension, values in (("result_class", predictions.target), ("calendar_month", pd.to_datetime(predictions.MatchDate).dt.strftime("%Y-%m")), ("home_team", predictions.HomeTeam), ("away_team", predictions.AwayTeam)):
                record["missingness"][dimension] = predictions.assign(dimension=values).groupby("dimension").family_available.agg(["count", "mean"]).reset_index().rename(columns={"mean": "coverage"}).to_dict("records")
            if not paired.empty:
                record.update(_comparison(paired, coverage=float(predictions.family_available.mean()), policy=policy, strict=strict, mapping_gate=mapping_gate))
                record["status"] = "evaluated"
                if name == "A0":
                    record["promotion_passed"] = False
                elif name != "A9" and record["promotion_passed"]:
                    report["recommended_families"].extend(families)
        report["ablations"][name] = record
    combined = report["ablations"]["A9"]
    report["release_gate"] = {"enabled_families": list(selected_families) if combined.get("promotion_passed") else [], "promotion_passed": bool(selected_families and combined.get("promotion_passed")), "reason": "Selected combination passed all gates" if combined.get("promotion_passed") else "Retain baseline until the selected combination passes every gate"}
    report["evidence_id"] = hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]
    return report, pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate paired PitchAPI feature-family ablations")
    parser.add_argument("--league", required=True)
    parser.add_argument("--data-dir", default="data_files")
    parser.add_argument("--output-dir", default="precomputed/model-audit")
    parser.add_argument("--retrospective", action="store_true")
    parser.add_argument("--selected-families", nargs="*", choices=FAMILIES, default=[])
    args = parser.parse_args(argv)
    root, destination = Path(args.data_dir), Path(args.output_dir)
    frame = read_frame(root / "combined_historical_data_with_calculations_new.csv")
    health = json.loads((root / "pitchapi_health.json").read_text()) if (root / "pitchapi_health.json").exists() else {}
    report, predictions = evaluate_pitchapi_families(frame, league_key=args.league, strict=not args.retrospective, mapping_gate=bool(health.get("mapping", {}).get("gate_passed")), selected_families=tuple(args.selected_families))
    atomic_json(destination / "pitchapi_ablation.json", report)
    write_frame(predictions, destination / "pitchapi_ablation_predictions.parquet")
    if report["release_gate"]["promotion_passed"]:
        configuration = FeatureFamilyConfig(args.league, tuple(report["release_gate"]["enabled_families"]), report["evidence_id"])
        atomic_json(root / "pitchapi_feature_config.json", configuration.as_dict())
    print(f"PitchAPI evaluation {report['evidence_id']}: promotion {report['release_gate']['promotion_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
