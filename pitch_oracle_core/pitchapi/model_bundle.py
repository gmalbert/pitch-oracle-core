"""Independent fitted baseline and promoted models with exact serving contracts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from pitch_oracle_core.features import FEATURE_POLICY_VERSION, chronological_partition_indices, no_odds_feature_columns, completed_future_rows
from pitch_oracle_core.features.families import FeatureFamilyConfig, feature_family
from pitch_oracle_core.predictions import FeatureContract, build_upcoming_feature_matrix
from pitch_oracle_core.pipelines import atomic_output
from .cache import atomic_json


def contract_fingerprint(contract: FeatureContract, *, league_key: str, families: tuple[str, ...], trained_through: str, trained_at: str) -> str:
    payload = {"version": contract.version, "league_key": league_key, "families": list(families), "trained_through": trained_through, "trained_at": trained_at, "feature_names": list(contract.feature_names), "imputation_values": dict(contract.imputation_values), "state_sources": {name: dict(source) for name, source in contract.state_sources.items()}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass
class ModelBundle:
    model: object
    contract: FeatureContract
    league_key: str
    families: tuple[str, ...]
    trained_through: str
    trained_at: str
    model_id: str
    fingerprint: str
    evidence_id: str | None = None
    schema_version: int = 1

    def validate(self, *, league_key: str) -> None:
        if self.schema_version != 1 or self.league_key != league_key or self.contract.version != FEATURE_POLICY_VERSION:
            raise ValueError("Model bundle has an incompatible league or schema")
        if self.fingerprint != contract_fingerprint(self.contract, league_key=self.league_key, families=self.families, trained_through=self.trained_through, trained_at=self.trained_at):
            raise ValueError("Model contract fingerprint mismatch")
        if getattr(self.model, "n_features_in_", None) != len(self.contract.feature_names):
            raise ValueError("Fitted model width disagrees with its contract")
        if not np.array_equal(np.asarray(self.model.classes_), [0, 1, 2]):
            raise ValueError("Fitted model requires all three outcome classes")
        if self.families and not self.evidence_id:
            raise ValueError("Promoted model lacks validation evidence")
        allowed = no_odds_feature_columns(pd.DataFrame(columns=self.contract.feature_names), enabled_families=self.families)
        if tuple(allowed) != self.contract.feature_names:
            raise ValueError("Model contract contains unapproved features")


def fit_bundle(frame: pd.DataFrame, *, league_key: str, families: tuple[str, ...] = (), evidence_id: str | None = None, model_id: str = "pitchapi-baseline-v1") -> ModelBundle:
    """Fit all preprocessing on the training period, calibration on a later period."""
    from models.no_odds_predictor import create_no_odds_classifier
    from pitch_oracle_core.model_audit import TemperatureScaledClassifier, fit_temperature

    frame = frame.sort_values("kickoff_utc" if "kickoff_utc" in frame else "MatchDate", kind="stable").copy()
    if not completed_future_rows(frame).empty:
        raise ValueError("Model training contains completed future matches")
    y = frame.FullTimeResult.map({"H": 0, "D": 1, "A": 2})
    if y.isna().any():
        raise ValueError("Model training requires completed outcomes")
    columns = frame[no_odds_feature_columns(frame, enabled_families=families)].select_dtypes(include=[np.number]).columns.tolist()
    if not columns:
        raise ValueError("Model training requires safe numeric features")
    x = frame[columns].replace([np.inf, -np.inf], np.nan)
    train, calibration, _test = chronological_partition_indices(frame.MatchDate, calibration_size=.2, test_size=.2)
    columns = [column for column in columns if feature_family(column) is None or x.iloc[train][column].notna().any()]
    x = x[columns]
    for family in families:
        if not any(feature_family(column) == family for column in columns):
            raise ValueError(f"Promoted family {family} lacks eligible training observations")
    if set(y.iloc[train].astype(int)) != {0, 1, 2}:
        raise ValueError("Training period requires all three outcome classes")
    means = x.iloc[train].mean().fillna(0)
    x = x.fillna(means)
    model = create_no_odds_classifier().fit(x.iloc[train].to_numpy(), y.iloc[train].astype(int).to_numpy())
    temperature = fit_temperature(model.predict_proba(x.iloc[calibration].to_numpy()), y.iloc[calibration].astype(int).to_numpy())
    model = TemperatureScaledClassifier(model, temperature)
    state_sources = {}
    for name in columns:
        # Provider state must come from the live mart, never a carried historical row.
        if feature_family(name) is not None:
            continue
        for home_prefix, away_prefix in (("Home", "Away"), ("home_", "away_")):
            if name.startswith(home_prefix):
                other = away_prefix + name[len(home_prefix):]
                home_column, away_column, role = name, other, "home"
            elif name.startswith(away_prefix):
                other = home_prefix + name[len(away_prefix):]
                home_column, away_column, role = other, name, "away"
            else:
                continue
            if other in columns:
                state_sources[name] = {"fixture_role": role, "home_history_column": home_column, "away_history_column": away_column}
            break
    contract = FeatureContract(FEATURE_POLICY_VERSION, tuple(columns), means.to_dict(), state_sources)
    through = pd.to_datetime(frame.MatchDate, utc=True).iloc[calibration].max().isoformat()
    trained_at = pd.Timestamp.now(tz="UTC").isoformat()
    bundle = ModelBundle(model=model, contract=contract, league_key=league_key, families=families, trained_through=through, trained_at=trained_at, model_id=model_id, fingerprint=contract_fingerprint(contract, league_key=league_key, families=families, trained_through=through, trained_at=trained_at), evidence_id=evidence_id)
    bundle.validate(league_key=league_key)
    return bundle


def train_model_bundles(frame: pd.DataFrame, *, configuration: FeatureFamilyConfig, models_dir: str | Path) -> dict:
    """Always fit a separate baseline; promoted features never alter its width."""
    root = Path(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    baseline = fit_bundle(frame, league_key=configuration.league_key)
    atomic_output(root / "pitchapi_baseline.pkl", lambda temporary: temporary.write_bytes(pickle.dumps(baseline)))
    records = {"baseline": {"model_id": baseline.model_id, "fingerprint": baseline.fingerprint, "feature_count": len(baseline.contract.feature_names)}}
    if configuration.enabled_families:
        promoted = fit_bundle(frame, league_key=configuration.league_key, families=configuration.enabled_families, evidence_id=configuration.evidence_id, model_id="pitchapi-promoted-v1")
        atomic_output(root / "pitchapi_promoted.pkl", lambda temporary: temporary.write_bytes(pickle.dumps(promoted)))
        records["promoted"] = {"model_id": promoted.model_id, "fingerprint": promoted.fingerprint, "feature_count": len(promoted.contract.feature_names), "families": list(promoted.families), "evidence_id": promoted.evidence_id}
    atomic_json(root / "pitchapi_models.json", {"schema_version": 1, "configuration": configuration.as_dict(), "models": records})
    return records


def load_bundle(path: str | Path, *, league_key: str) -> ModelBundle:
    with Path(path).open("rb") as stream:
        bundle = pickle.load(stream)
    if not isinstance(bundle, ModelBundle):
        raise ValueError("Unexpected model bundle type")
    bundle.validate(league_key=league_key)
    return bundle


def predict_with_fallback(historical: pd.DataFrame, upcoming: pd.DataFrame, *, baseline: ModelBundle, promoted: ModelBundle | None = None, capability_health: dict | None = None, as_of: pd.Timestamp | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Select a fitted model per fixture and record every baseline fallback reason."""
    baseline.validate(league_key=baseline.league_key)
    if baseline.families:
        raise ValueError("Fallback must be an independently trained baseline")
    if promoted is not None:
        promoted.validate(league_key=baseline.league_key)
    required_capabilities = {"xg": "shots", "shot_profile": "shots", "advanced_team": "advanced_team", "style": "advanced_team", "player_strength": "advanced_player", "predicted_lineup": "predicted_lineups", "confirmed_lineup": "confirmed_lineups", "goalkeeper": "advanced_player"}
    probabilities, metadata = [], []
    for _, fixture in upcoming.iterrows():
        cutoff_value = as_of if as_of is not None else fixture.get("as_of")
        cutoff = pd.Timestamp(cutoff_value) if cutoff_value is not None else pd.Timestamp.now(tz="UTC")
        if pd.isna(cutoff) or cutoff.tzinfo is None or pd.Timestamp(baseline.trained_at) > cutoff:
            raise ValueError("Baseline model was not available at the forecast cutoff")
        reason = "no_promoted_model" if promoted is None else None
        if promoted is not None:
            cutoff = pd.Timestamp(as_of if as_of is not None else fixture.get("as_of"))
            kickoff = pd.Timestamp(fixture.get("kickoff_utc"))
            if pd.isna(cutoff) or cutoff.tzinfo is None or pd.isna(kickoff) or kickoff.tzinfo is None or cutoff >= kickoff:
                raise ValueError("Promoted forecasts require an explicit pre-kickoff as_of")
            lineage = pd.to_datetime(fixture.get("feature_observed_at"), utc=True, errors="coerce")
            if pd.isna(lineage) or lineage > cutoff:
                reason = "missing_or_late_feature_lineage"
            if pd.Timestamp(promoted.trained_at) > cutoff or pd.Timestamp(promoted.trained_through) >= cutoff:
                reason = "model_not_available_at_cutoff"
            health = capability_health or {}
            checked = pd.to_datetime(health.get("checked_at"), utc=True, errors="coerce")
            if health.get("mapping", {}).get("gate_passed") is not True:
                reason = "fixture_mapping_not_validated"
            if pd.isna(checked) or checked > cutoff or cutoff - checked > pd.Timedelta(hours=24):
                reason = "capability_audit_stale_or_missing"
            if health.get("last_run_status") == "unavailable":
                reason = "provider_unavailable"
            for family in promoted.families:
                name = required_capabilities[family]
                capability = (capability_health or {}).get("capabilities", {}).get(name, {})
                # Missing predicted data remains operational: this fixture uses the baseline.
                if capability.get("status") not in {"available", "degraded"}:
                    reason = f"{name}_{capability.get('status', 'unknown')}"
                    break
                required = [column for column in promoted.contract.feature_names if feature_family(column) == family and "coverage" not in column and "is_confirmed" not in column]
                if required and not any(pd.notna(fixture.get(column)) for column in required):
                    reason = f"{family}_missing_required_features"
                    break
                for side in ("home", "away"):
                    side_required = [column for column in required if column.startswith(f"{side}_")]
                    if side_required and not any(pd.notna(fixture.get(column)) for column in side_required):
                        reason = f"{family}_{side}_missing_required_features"
                    if family in {"predicted_lineup", "confirmed_lineup"}:
                        observed = pd.to_datetime(fixture.get(f"{side}_lineup_observed_at"), utc=True, errors="coerce")
                        maximum_age = pd.Timedelta(hours=6) if family == "predicted_lineup" else pd.Timedelta(minutes=75)
                        if pd.isna(observed) or observed > cutoff or cutoff - observed > maximum_age:
                            reason = f"{family}_{side}_stale_or_missing"
        selected = baseline if reason is not None else promoted
        assert selected is not None
        matrix = build_upcoming_feature_matrix(historical, pd.DataFrame([fixture]), selected.contract, as_of=as_of)
        output = np.asarray(selected.model.predict_proba(matrix), dtype=float)
        if output.shape != (1, 3) or not np.isfinite(output).all() or (output < 0).any() or not np.isclose(output.sum(), 1):
            raise ValueError("Model returned invalid 1X2 probabilities")
        probabilities.append(output[0])
        metadata.append({"fixture_id": fixture.get("fixture_id"), "model_id": selected.model_id, "model_fingerprint": selected.fingerprint, "fallback_used": reason is not None, "fallback_reason": reason, "feature_families": ",".join(selected.families)})
    return np.asarray(probabilities).reshape(-1, 3), pd.DataFrame(metadata)


def main(argv: list[str] | None = None) -> int:
    import argparse
    from .storage import read_frame

    parser = argparse.ArgumentParser(description="Train independent PitchAPI baseline and promoted model bundles")
    parser.add_argument("--league", required=True)
    parser.add_argument("--data-dir", default="data_files")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--historical-file", default="combined_historical_data_with_calculations_new.csv")
    args = parser.parse_args(argv)
    configuration = FeatureFamilyConfig.load(Path(args.data_dir) / "pitchapi_feature_config.json", league_key=args.league)
    records = train_model_bundles(read_frame(Path(args.data_dir) / args.historical_file), configuration=configuration, models_dir=args.models_dir)
    print(f"Trained {len(records)} independent model bundles for {args.league}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
