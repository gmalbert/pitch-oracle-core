"""Capability-specific coverage, freshness, latency, and missingness audits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd

from .cache import atomic_json
from .contracts import INTEGRATION_SCHEMA_VERSION, utc_timestamp
from .storage import read_frame
from .revisions import complete_responses


def _latest(frame: pd.DataFrame, keys: list[str], timestamp: str = "observed_at") -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result[timestamp] = pd.to_datetime(result[timestamp], utc=True, errors="raise")
    return result.sort_values(timestamp, kind="stable").drop_duplicates(keys, keep="last")


def capability_from_frame(*, name: str, expected_ids: set, observed_ids: set, observed_at: str | None, now: datetime, maximum_age_hours: float | None = None, failed: bool = False) -> dict:
    eligible = expected_ids.intersection(observed_ids)
    coverage = len(eligible) / len(expected_ids) if expected_ids else 0.0
    status = "unavailable" if not eligible else "available" if coverage >= .999 else "degraded"
    if failed:
        status = "degraded" if eligible else "failed"
    elif eligible and maximum_age_hours is not None and observed_at is not None:
        if (now - utc_timestamp(observed_at)).total_seconds() > maximum_age_hours * 3600:
            status = "stale"
    return {"name": name, "status": status, "coverage": coverage, "expected": len(expected_ids), "available": len(eligible), "observed_at": observed_at, "message": "No eligible fixtures in the audit window" if not expected_ids else "Latest refresh failed; retained observations remain available" if failed else ""}


def _observation_time(frame: pd.DataFrame, timestamp: str = "observed_at") -> str | None:
    if frame.empty or timestamp not in frame:
        return None
    values = pd.to_datetime(frame[timestamp], utc=True, errors="coerce").dropna()
    return values.max().isoformat() if not values.empty else None


def build_coverage_report(data_dir: str | Path, *, league_key: str, now: datetime | None = None, season: str | None = None) -> dict:
    now = utc_timestamp(now or datetime.now(timezone.utc))
    data_dir = Path(data_dir)
    matches = read_frame(data_dir / "pitchapi_matches.csv")
    audit = read_frame(data_dir / "pitchapi_fixture_audit.csv")
    summary = read_frame(data_dir / "pitchapi_match_shot_features.csv")
    advanced = read_frame(data_dir / "pitchapi_advanced_team.parquet")
    players = read_frame(data_dir / "pitchapi_player_match.parquet")
    lineups = read_frame(data_dir / "pitchapi_lineup_snapshots.parquet")
    network = read_frame(data_dir / "pitchapi_network.parquet")
    momentum = read_frame(data_dir / "pitchapi_momentum.parquet")
    revisions = read_frame(data_dir / "pitchapi_response_revisions.parquet")
    advanced, players, network, momentum = [complete_responses(frame, revisions, artifact=name) for frame, name in ((advanced, "pitchapi_advanced_team"), (players, "pitchapi_player_match"), (network, "pitchapi_network"), (momentum, "pitchapi_momentum"))]
    run_path = data_dir / "pitchapi_provider_run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    if season is not None and not matches.empty:
        matches = matches.loc[matches.season.astype(str) == season].copy()
        provider_ids = set(matches.match_id.astype(str))
        if not audit.empty:
            audit = audit.loc[audit.provider_match_id.astype(str).isin(provider_ids)]
        canonical_ids = set(matches.fixture_id.dropna().astype(str))
        summary, advanced, players, lineups, network, momentum = [frame.loc[frame.fixture_id.astype(str).isin(canonical_ids)] if not frame.empty else frame for frame in (summary, advanced, players, lineups, network, momentum)]
    report = {"provider": "pitchapi", "schema_version": INTEGRATION_SCHEMA_VERSION, "league_key": league_key, "checked_at": now.isoformat(), "last_run_status": run.get("status", "unavailable"), "capabilities": {}, "mapping": {"gate_passed": False, "coverage": 0.0, "expected": 0, "mapped": 0}, "latency_minutes": {}, "lineup_lead_minutes": {}, "missingness": {}}
    if not audit.empty:
        counts = audit.status.value_counts().to_dict()
        total = len(audit.loc[audit.status.ne("outside_source_window")])
        coverage = counts.get("mapped", 0) / total if total else 0.0
        unresolved = audit.loc[~audit.status.isin(["mapped", "outside_source_window"])].astype(object)
        unresolved = unresolved.where(pd.notna(unresolved), None)
        report["mapping"] = {"expected": total, "mapped": int(counts.get("mapped", 0)), "coverage": coverage, "outside_source_window": int(counts.get("outside_source_window", 0)), "gate_passed": coverage >= .995 and counts.get("ambiguous", 0) == 0 and counts.get("reversed", 0) == 0, "unresolved": unresolved.to_dict("records")}
    if matches.empty:
        report["capabilities"] = {name: capability_from_frame(name=name, expected_ids=set(), observed_ids=set(), observed_at=None, now=now, failed=run.get("status") == "degraded") for name in ("schedules", "predicted_lineups", "confirmed_lineups", "shots", "advanced_team", "advanced_player", "network", "momentum")}
        return report
    mapped = matches.loc[matches.fixture_id.notna()].copy()
    mapped["kickoff_utc"] = pd.to_datetime(mapped.kickoff_utc, utc=True, errors="raise")
    historical = mapped.loc[mapped.status == "finished"]
    upcoming = mapped.loc[(mapped.kickoff_utc > now) & (mapped.kickoff_utc <= now + pd.Timedelta(hours=48))]
    completed_ids = set(historical.fixture_id.astype(str))
    upcoming_ids = set(upcoming.fixture_id.astype(str))
    errors = run.get("errors", [])
    failing_endpoints = {item.get("endpoint") for item in errors}
    listing_failed = any(item.get("endpoint") is None for item in errors)
    report["capabilities"]["schedules"] = capability_from_frame(name="schedules", expected_ids=set(matches.match_id.astype(str)), observed_ids=set(mapped.match_id.astype(str)), observed_at=_observation_time(matches), now=now, maximum_age_hours=12, failed=listing_failed)
    for name, frame, endpoint in (("shots", summary, "shots"), ("advanced_team", advanced, "advanced"), ("advanced_player", players, "advanced/players"), ("network", network, "advanced/network"), ("momentum", momentum, "momentum")):
        ids = set(frame.fixture_id.astype(str)) if not frame.empty else set()
        expected = completed_ids
        capability = capability_from_frame(name=name, expected_ids=expected, observed_ids=ids, observed_at=_observation_time(frame), now=now, failed=listing_failed or endpoint in failing_endpoints)
        overdue = historical.loc[~historical.fixture_id.isin(ids) & (historical.kickoff_utc + pd.Timedelta(hours=2) < now - pd.Timedelta(hours=48))]
        if not overdue.empty and capability["status"] not in {"failed", "unavailable"}:
            capability["status"] = "stale"
            capability["message"] = f"{len(overdue)} completed fixtures remain missing beyond the 48-hour delivery window"
        if name == "advanced_team" and not frame.empty:
            latest = _latest(frame, ["fixture_id", "team_id"])
            pairs = set(zip(latest.fixture_id.astype(str), latest.team_id.astype(str)))
            expected_pairs = {(str(row.fixture_id), str(getattr(row, f"{side}_team_id"))) for row in historical.itertuples() for side in ("home", "away")}
            capability["team_coverage"] = len(pairs.intersection(expected_pairs)) / len(expected_pairs) if expected_pairs else 0.0
            capability["coverage"] = capability["team_coverage"]
            if capability["status"] == "available" and capability["coverage"] < .999:
                capability["status"] = "degraded"
        report["capabilities"][name] = capability
        if not historical.empty:
            availability = historical.assign(available=historical.fixture_id.isin(ids))
            for dimension, values in (
                ("season", availability.season),
                ("calendar_month", availability.kickoff_utc.dt.strftime("%Y-%m")),
                ("result_class", availability.apply(lambda row: "unknown" if pd.isna(row.score_home) or pd.isna(row.score_away) else "home" if row.score_home > row.score_away else "away" if row.score_home < row.score_away else "draw", axis=1)),
                ("home_team", availability.home_team_id), ("away_team", availability.away_team_id),
            ):
                report["missingness"].setdefault(name, {})[dimension] = availability.assign(dimension=values).groupby("dimension", dropna=False).available.agg(["count", "mean"]).reset_index().rename(columns={"mean": "coverage"}).to_dict("records")
    for status in ("predicted", "confirmed"):
        eligible = pd.DataFrame()
        if not lineups.empty:
            rows = lineups.copy()
            rows["snapshot_at"] = pd.to_datetime(rows.snapshot_at, utc=True, errors="raise")
            rows["kickoff_utc"] = pd.to_datetime(rows.kickoff_utc, utc=True, errors="raise")
            eligible = rows.loc[(rows.snapshot_at < rows.kickoff_utc) & (rows.snapshot_at <= now) & (rows.lineup_status == status)]
        complete = set()
        lead_times = []
        if not eligible.empty:
            counts = eligible.loc[eligible.is_starter.eq(True)].groupby(["fixture_id", "team_id", "snapshot_id"]).player_id.nunique()
            complete = {(str(fixture), str(team)) for (fixture, team, snapshot), count in counts.items() if count == 11}
            first = eligible.groupby(["fixture_id", "team_id"]).agg({"snapshot_at": "min", "kickoff_utc": "first"})
            lead_times = ((first.kickoff_utc - first.snapshot_at).dt.total_seconds() / 60).tolist()
        due = upcoming if status == "predicted" else upcoming.loc[upcoming.kickoff_utc <= now + pd.Timedelta(minutes=90)]
        expected_pairs = {(str(row.fixture_id), str(getattr(row, f"{side}_team_id"))) for row in due.itertuples() for side in ("home", "away")}
        capability = capability_from_frame(name=f"{status}_lineups", expected_ids=expected_pairs, observed_ids=complete, observed_at=_observation_time(eligible, "snapshot_at"), now=now, maximum_age_hours=6 if status == "predicted" else 1.25, failed=listing_failed or "lineups" in failing_endpoints)
        # Confirmed-lineup freshness is a polling outcome, not expiry of an unchanged official XI.
        if capability["status"] == "stale" and status == "confirmed":
            capability["message"] = "Latest lineup observation is older than the near-kickoff freshness target"
        report["capabilities"][f"{status}_lineups"] = capability
        report["lineup_lead_minutes"][status] = {"samples": len(lead_times), "median": float(pd.Series(lead_times).median()) if lead_times else None}
    if not summary.empty:
        first = summary.assign(observed_at=pd.to_datetime(summary.observed_at, utc=True)).groupby("fixture_id", as_index=False).observed_at.min()
        delay = first.merge(historical[["fixture_id", "kickoff_utc"]], on="fixture_id", validate="one_to_one")
        minutes = (delay.observed_at - (delay.kickoff_utc + pd.Timedelta(minutes=120))).dt.total_seconds() / 60
        # Backfills do not measure actual provider publication latency.
        operational = minutes.loc[(minutes >= 0) & (minutes <= 48 * 60)]
        report["latency_minutes"] = {"method": "first observation minus kickoff+120m; operational approximation", "samples": len(operational), "excluded_backfills": int((minutes > 48 * 60).sum()), **{label: float(operational.quantile(q)) if len(operational) else None for label, q in (("median", .5), ("p90", .9), ("p95", .95), ("max", 1.0))}}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit capability-specific PitchAPI coverage")
    parser.add_argument("--league", required=True)
    parser.add_argument("--season", help="Restrict capability and missingness reports to this season")
    parser.add_argument("--data-dir", default="data_files")
    parser.add_argument("--output", help="Default: DATA_DIR/pitchapi_health.json and the provider-health compatibility report")
    args = parser.parse_args(argv)
    report = build_coverage_report(args.data_dir, league_key=args.league, season=args.season)
    atomic_json(Path(args.output) if args.output else Path(args.data_dir) / "pitchapi_health.json", report)
    if args.output is None:
        atomic_json(Path("precomputed/provider-health/pitchapi.json"), report)
    print(f"PitchAPI audit: {len(report['capabilities'])} capabilities, mapping gate {report['mapping'].get('gate_passed', False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
