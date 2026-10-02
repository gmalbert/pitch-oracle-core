"""Hourly immutable forecasts; closing references an actual successful issue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pitch_oracle_core.features.forecast_inputs import build_forecast_inputs
from pitch_oracle_core.features.families import FeatureFamilyConfig
from pitch_oracle_core.leagues import get_league_config
from pitch_oracle_core.players.lineup_snapshots import latest_eligible_lineup, lineup_status
from pitch_oracle_core.pitchapi.normalize import boolean
from .cache import atomic_json
from .coverage import build_coverage_report
from .ingest import refresh_pitchapi
from .model_bundle import load_bundle, predict_with_fallback
from .storage import read_frame, append_revisions, write_frame


def lineup_signature(lineup: pd.DataFrame) -> str:
    """Ignore capture time and player names; include material XI/formation changes."""
    if lineup.empty:
        return "none"
    fields = [column for column in ("player_id", "is_starter", "position_slot", "position", "formation", "lineup_status") if column in lineup]
    frame = lineup[fields].copy().sort_values("player_id", kind="stable").astype(object)
    frame = frame.where(pd.notna(frame), None)
    if "is_starter" in frame:
        frame["is_starter"] = frame.is_starter.map(boolean)
    return hashlib.sha256(json.dumps(frame.to_dict("records"), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_forecast_ledger(ledger: pd.DataFrame) -> None:
    required = {"fixture_id", "issued_at", "kickoff_utc", "model_id", "model_fingerprint", "p_home", "p_draw", "p_away", "revision_label"}
    if required.difference(ledger):
        raise ValueError(f"Forecast ledger misses {sorted(required.difference(ledger))}")
    if ledger.empty:
        return
    for field in ("issued_at", "kickoff_utc"):
        if any(pd.Timestamp(value).tzinfo is None for value in ledger[field]):
            raise ValueError("Forecast timestamps must be timezone-aware")
    issue = pd.to_datetime(ledger.issued_at, utc=True, errors="raise")
    kickoff = pd.to_datetime(ledger.kickoff_utc, utc=True, errors="raise")
    if issue.isna().any() or kickoff.isna().any() or (issue >= kickoff).any():
        raise ValueError("Forecast ledger contains an issue at or after kickoff")
    if ledger[["fixture_id", "model_id", "model_fingerprint"]].isna().any().any() or ledger.duplicated(["fixture_id", "issued_at", "model_fingerprint"]).any():
        raise ValueError("Forecast identity must be present and unique")
    probabilities = ledger[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Forecast probabilities must be normalized")


def closing_forecasts(ledger: pd.DataFrame, *, as_of: pd.Timestamp, current_fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
    """No new inference or issue timestamp is created after kickoff."""
    if ledger.empty:
        return ledger.copy()
    validate_forecast_ledger(ledger)
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("Closing selection requires a timezone-aware cutoff")
    frame = ledger.copy()
    frame["issued_at"] = pd.to_datetime(frame.issued_at, utc=True)
    frame["kickoff_utc"] = pd.to_datetime(frame.kickoff_utc, utc=True)
    frame = frame.loc[frame.issued_at <= cutoff].sort_values("issued_at", kind="stable").drop_duplicates("fixture_id", keep="last")
    if current_fixtures is not None and not current_fixtures.empty:
        known = current_fixtures.drop_duplicates("fixture_id", keep="last").set_index("fixture_id").kickoff_utc
        current = pd.to_datetime(frame.fixture_id.map(known), utc=True, errors="raise")
        # A stale forecast for the old date of a rescheduled match is not its close.
        frame = frame.loc[current.isna() | frame.kickoff_utc.eq(current)].copy()
    frame = frame.loc[frame.kickoff_utc <= cutoff].copy()
    frame["issued_stage"] = frame.revision_label
    frame["revision_label"] = "closing"
    frame["closing_selected_at"] = cutoff.isoformat()
    return frame


def replay_forecast(ledger: pd.DataFrame, *, fixture_id: str, as_of: pd.Timestamp) -> pd.Series | None:
    if ledger.empty:
        return None
    validate_forecast_ledger(ledger)
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("Replay requires a timezone-aware cutoff")
    eligible = ledger.loc[ledger.fixture_id.astype(str).eq(str(fixture_id)) & (pd.to_datetime(ledger.issued_at, utc=True) <= cutoff)]
    return eligible.sort_values("issued_at", kind="stable").iloc[-1] if not eligible.empty else None


def capture_hourly_forecasts(inputs: pd.DataFrame, snapshots: pd.DataFrame, *, as_of: pd.Timestamp, destination: str | Path, predictor: Callable[[pd.DataFrame], tuple[np.ndarray, pd.DataFrame]]) -> tuple[pd.DataFrame, list[dict]]:
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("Hourly forecasts require a timezone-aware cutoff")
    ledger = read_frame(destination)
    errors = []
    for _, fixture in inputs.iterrows():
        kickoff = pd.Timestamp(fixture.kickoff_utc)
        if kickoff.tzinfo is None or cutoff >= kickoff:
            continue
        try:
            previous = replay_forecast(ledger, fixture_id=str(fixture.fixture_id), as_of=cutoff) if not ledger.empty else None
            row = {"fixture_id": str(fixture.fixture_id), "kickoff_utc": kickoff.isoformat(), "issued_at": cutoff.isoformat(), "as_of": cutoff.isoformat(), "revision_label": "initial" if previous is None else "hourly"}
            confirmed_change = False
            for side in ("home", "away"):
                lineup = latest_eligible_lineup(snapshots, fixture_id=str(fixture.fixture_id), team_id=str(fixture[f"{side}_team_id"]), as_of=cutoff, require_complete=True)
                status = lineup_status(lineup)
                signature = lineup_signature(lineup)
                row[f"{side}_lineup_status"] = status
                row[f"{side}_lineup_signature"] = signature
                row[f"{side}_lineup_observed_at"] = pd.to_datetime(lineup.snapshot_at, utc=True).max().isoformat() if not lineup.empty else None
                if previous is not None and status == "confirmed" and signature != previous.get(f"{side}_lineup_signature"):
                    confirmed_change = True
            if previous is not None:
                history = ledger.loc[ledger.fixture_id.astype(str).eq(str(fixture.fixture_id))]
                if confirmed_change:
                    row["revision_label"] = "lineup"
                elif kickoff != pd.Timestamp(previous.kickoff_utc):
                    row["revision_label"] = "schedule_update"
                elif kickoff - cutoff <= pd.Timedelta(hours=24) and not history.revision_label.eq("24_hour").any():
                    row["revision_label"] = "24_hour"
            probabilities, metadata = predictor(pd.DataFrame([fixture]))
            if probabilities.shape != (1, 3) or len(metadata) != 1:
                raise ValueError("Hourly predictor returned an invalid result shape")
            row.update({column: metadata.iloc[0][column] for column in metadata if column != "fixture_id"})
            if previous is not None and pd.Timestamp(previous.issued_at) == cutoff and previous.model_fingerprint == row.get("model_fingerprint"):
                row["revision_label"] = previous.revision_label
            row.update(zip(("p_home", "p_draw", "p_away"), map(float, probabilities[0])))
            row["feature_observed_at"] = fixture.get("feature_observed_at")
            revision = pd.DataFrame([row])
            validate_forecast_ledger(revision)
            ledger = append_revisions(revision, destination, keys=["fixture_id", "issued_at", "model_fingerprint"])
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors.append({"fixture_id": str(fixture.fixture_id), "code": type(exc).__name__, "message": str(exc)})
    return ledger, errors


def refresh_lineup_forecasts(*, league_key: str, data_dir: str | Path = "data_files", models_dir: str | Path = "models", now: pd.Timestamp | None = None, refresh_provider: bool = True) -> dict:
    cutoff = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    config, root = get_league_config(league_key), Path(data_dir)
    if refresh_provider:
        refresh_pitchapi(config, output_dir=root, with_shots=False, with_lineups=True, status="upcoming", now=cutoff.to_pydatetime())
    health = build_coverage_report(root, league_key=league_key, now=cutoff.to_pydatetime())
    atomic_json(root / "pitchapi_health.json", health)
    history = read_frame(root / "combined_historical_data_with_calculations_new.csv")
    upcoming = read_frame(root / "upcoming_fixtures.csv")
    if upcoming.empty:
        existing = read_frame(root / "pitchapi_forecast_revisions.parquet")
        if not existing.empty:
            write_frame(closing_forecasts(existing, as_of=cutoff), root / "pitchapi_closing_forecasts.parquet")
        report = {"checked_at": cutoff.isoformat(), "league_key": league_key, "status": "no_upcoming_fixtures", "errors": []}
        atomic_json(root / "pitchapi_forecast_run.json", report)
        return report
    # Canonical preparation validates timestamps; filter before calling its pre-match builder.
    from pitch_oracle_core.fixtures.canonical import canonical_fixture_frame
    upcoming = canonical_fixture_frame(upcoming, config, input_timezone=config.sources.upcoming_timezone)
    upcoming = upcoming.loc[(upcoming.kickoff_lower_bound_utc > cutoff) & (upcoming.kickoff_utc <= cutoff + pd.Timedelta(days=7)) & ~upcoming.status.isin(["cancelled", "canceled", "postponed"])]
    feature_failure = False
    try:
        inputs = build_forecast_inputs(history, upcoming, config=config, as_of=cutoff, data_dir=root)
    except (ValueError, KeyError, OSError):
        feature_failure = True
        inputs = build_forecast_inputs(history, upcoming, config=config, as_of=cutoff)
    baseline = load_bundle(Path(models_dir) / "pitchapi_baseline.pkl", league_key=league_key)
    configuration = FeatureFamilyConfig.load(root / "pitchapi_feature_config.json", league_key=league_key)
    promoted = None
    if configuration.enabled_families and not feature_failure:
        try:
            promoted = load_bundle(Path(models_dir) / "pitchapi_promoted.pkl", league_key=league_key)
            if promoted.families != configuration.enabled_families or promoted.evidence_id != configuration.evidence_id:
                promoted = None
        except (OSError, ValueError):
            pass
    def predict(frame):
        probabilities, metadata = predict_with_fallback(history, frame, baseline=baseline, promoted=promoted, capability_health=health, as_of=cutoff)
        if feature_failure:
            metadata["fallback_reason"] = "provider_feature_build_failed"
        return probabilities, metadata
    ledger, errors = capture_hourly_forecasts(inputs, read_frame(root / "pitchapi_lineup_snapshots.parquet"), as_of=cutoff, destination=root / "pitchapi_forecast_revisions.parquet", predictor=predict)
    if not ledger.empty:
        write_frame(closing_forecasts(ledger, as_of=cutoff, current_fixtures=upcoming), root / "pitchapi_closing_forecasts.parquet")
    report = {"checked_at": cutoff.isoformat(), "league_key": league_key, "status": "degraded" if errors or feature_failure else "available", "fixtures": len(inputs), "provider_feature_failure": feature_failure, "errors": errors}
    atomic_json(root / "pitchapi_forecast_run.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh hourly lineup snapshots and immutable forecasts")
    parser.add_argument("--league", required=True)
    parser.add_argument("--data-dir", default="data_files")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--offline", action="store_true", help="Use stored observations without requesting the provider")
    args = parser.parse_args(argv)
    report = refresh_lineup_forecasts(league_key=args.league, data_dir=args.data_dir, models_dir=args.models_dir, refresh_provider=not args.offline)
    print(f"Hourly forecast refresh: {report['status']}, {len(report['errors'])} errors")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
