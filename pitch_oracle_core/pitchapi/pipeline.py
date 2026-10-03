"""Daily enrichment and hourly capture commands for every league consumer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pitch_oracle_core.features.historical import prepare_historical_features
from pitch_oracle_core.features.families import FeatureFamilyConfig
from .artifacts import publish_index
from .cache import atomic_json
from .coverage import build_coverage_report
from .forecast_lifecycle import refresh_lineup_forecasts
from .ingest import refresh_pitchapi
from .model_bundle import train_model_bundles
from .storage import read_frame


def daily(*, league_key, data_dir="data_files", historical_file="combined_historical_data_with_calculations_new.csv", seasons=None, refresh=False, request_budget=500):
    root = Path(data_dir)
    refresh_pitchapi(league_key, output_dir=root, seasons=seasons, with_shots=True, with_advanced=True, with_players=True, with_lineups=True, with_momentum=True, with_network=True, with_heatmaps=True, refresh=refresh, request_budget=request_budget)
    atomic_json(root / "pitchapi_health.json", build_coverage_report(root, league_key=league_key))
    # The package import avoids shadowing by standalone consumers' legacy modules.
    try:
        prepare_historical_features(league_key=league_key, source=root / "combined_historical_data.csv", destination=root / historical_file, xg_source=root / "pitchapi_match_xg.csv", pitchapi_data_dir=root)
    except (ValueError, KeyError, OSError) as exc:
        # Retry baseline preparation; invalid primary data must still fail this job.
        prepare_historical_features(league_key=league_key, source=root / "combined_historical_data.csv", destination=root / historical_file)
        atomic_json(root / "pitchapi_feature_build.json", {"status": "degraded", "code": type(exc).__name__, "fallback": "baseline_history"})
    publish_index(Path.cwd(), league_key, data_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["daily", "train", "hourly", "index"])
    parser.add_argument("--league", default=os.getenv("PITCH_ORACLE_LEAGUE"), required=not os.getenv("PITCH_ORACLE_LEAGUE"))
    parser.add_argument("--data-dir", default="data_files")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--historical-file", default="combined_historical_data_with_calculations_new.csv")
    parser.add_argument("--seasons", nargs="*", help="Explicit seasons for a historical backfill")
    parser.add_argument("--all-seasons", action="store_true", help="Backfill every provider season, newest first; requires primary historical identities")
    parser.add_argument("--request-budget", type=int, default=500, help="Maximum new analytics requests per run; saved responses are reused on continuation")
    parser.add_argument("--refresh", action="store_true", help="Manually refresh completed responses after the seven-day correction period")
    args = parser.parse_args(argv)
    if args.mode == "daily":
        if args.all_seasons and args.seasons:
            parser.error("Choose --all-seasons or --seasons")
        daily(league_key=args.league, data_dir=args.data_dir, historical_file=args.historical_file, seasons=["all"] if args.all_seasons else args.seasons, refresh=args.refresh, request_budget=args.request_budget)
    elif args.mode == "train":
        config = FeatureFamilyConfig.load(Path(args.data_dir) / "pitchapi_feature_config.json", league_key=args.league)
        train_model_bundles(read_frame(Path(args.data_dir) / args.historical_file), configuration=config, models_dir=args.models_dir)
    elif args.mode == "hourly":
        report = refresh_lineup_forecasts(league_key=args.league, data_dir=args.data_dir, models_dir=args.models_dir, historical_file=args.historical_file)
        publish_index(Path.cwd(), args.league, args.data_dir)
        print(f"Hourly capture: {report['status']}; {len(report['errors'])} forecast errors")
        return 1 if report["errors"] else 0
    else:
        publish_index(Path.cwd(), args.league, args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
