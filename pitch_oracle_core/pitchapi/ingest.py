"""One league refresh: optional failures are recorded without erasing valid data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Callable
import pandas as pd

from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.fixtures.canonical import canonical_fixture_frame
from pitch_oracle_core.fixtures.provider_mapping import reconcile_fixtures
from pitch_oracle_core.leagues import get_league_config
from pitch_oracle_core.features.shot_profile import build_match_shot_features
from pitch_oracle_core.features import completed_match_rows
from .cache import ObservationCache, atomic_json
from .client import PitchAPIClient, PitchAPIError
from .contracts import INTEGRATION_SCHEMA_VERSION, utc_timestamp
from .normalize import normalize_matches, normalize_shots, normalize_team_data, normalize_player_data, normalize_lineups, normalize_network, normalize_heatmaps
from .storage import read_frame, write_frame, append_revisions

ARTIFACTS = {
    "pitchapi_matches": ".csv", "pitchapi_shots": ".parquet",
    "pitchapi_match_xg": ".csv", "pitchapi_match_shot_features": ".csv",
    "pitchapi_advanced_team": ".parquet", "pitchapi_player_match": ".parquet",
    "pitchapi_momentum": ".parquet", "pitchapi_network": ".parquet",
    "pitchapi_heatmaps": ".parquet", "pitchapi_lineup_snapshots": ".parquet",
}


def pitchapi_league_id(config: LeagueConfig) -> str:
    if not config.sources.pitchapi or not config.sources.pitchapi_league_id:
        raise ValueError(f"PitchAPI is not configured for {config.key}")
    return config.sources.pitchapi_league_id


def load_canonical_fixtures(data_dir: Path, config: LeagueConfig) -> pd.DataFrame:
    frames = []
    # Use raw historical identities shared with prepare_model_data, not an older enumerated model export.
    for name in ("combined_historical_data.csv", "upcoming_fixtures.csv"):
        path = data_dir / name
        if path.exists():
            source = read_frame(path)
            if name == "combined_historical_data.csv" and not source.empty:
                source = completed_match_rows(source, date_column="Date" if "Date" in source else "MatchDate", result_column="FTR" if "FTR" in source else "FullTimeResult")
            if not source.empty:
                frames.append(canonical_fixture_frame(source, config, input_timezone=config.sources.historical_timezone if name == "combined_historical_data.csv" else config.sources.upcoming_timezone))
    if not frames:
        raise ValueError("Canonical historical/upcoming fixtures are required before PitchAPI mapping")
    result = pd.concat(frames, ignore_index=True)
    # Historical and upcoming caches can contain the same fixture after it completes.
    duplicate = result.duplicated(["kickoff_utc", "home_team_id", "away_team_id"], keep="first")
    return result.loc[~duplicate].reset_index(drop=True)


def refresh_pitchapi(
    league: LeagueConfig | str = "epl", *, seasons: list[str] | None = None,
    output_dir: str | Path | None = None, canonical: pd.DataFrame | None = None,
    api_key: str | None = None, client: PitchAPIClient | None = None,
    with_shots: bool = True, with_advanced: bool = False, with_players: bool = False,
    with_lineups: bool = False, with_momentum: bool = False, with_network: bool = False,
    with_heatmaps: bool = False, status: str = "all", optional: bool = True,
    refresh: bool = False, now: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    config = get_league_config(league) if isinstance(league, str) else league
    league_id = pitchapi_league_id(config)
    now = utc_timestamp(now or datetime.now(timezone.utc))
    data_dir = Path(output_dir or os.getenv("PITCH_ORACLE_DATA_DIR", config.data_dir_name))
    data_dir.mkdir(parents=True, exist_ok=True)
    frames = {name: read_frame(data_dir / f"{name}{extension}") for name, extension in ARTIFACTS.items()}
    run = {"provider": "pitchapi", "schema_version": INTEGRATION_SCHEMA_VERSION, "league_key": config.key, "checked_at": now.isoformat(), "status": "available", "errors": [], "capabilities": {}}
    health_path = data_dir / "pitchapi_provider_run.json"

    def failure(code: str, *, match_id: str | None = None, endpoint: str | None = None):
        run["errors"].append({"code": code, "match_id": match_id, "endpoint": endpoint})
        run["status"] = "degraded"

    key = api_key or os.getenv("PITCH_API_KEY")
    if client is None and not key:
        run["status"] = "unavailable"
        failure("MISSING_CREDENTIAL")
        run["status"] = "unavailable"
        atomic_json(health_path, run)
        if not optional:
            raise PitchAPIError("MISSING_CREDENTIAL", "PITCH_API_KEY is not configured")
        return frames
    client = client or PitchAPIClient(key)
    cache = ObservationCache(data_dir / "pitchapi_cache")
    try:
        canonical = canonical if canonical is not None else load_canonical_fixtures(data_dir, config)
        if seasons is None:
            catalogue = client.leagues()
            record = next((item for item in catalogue if item["id"] == league_id), None)
            if record is None:
                raise PitchAPIError("LEAGUE_NOT_FOUND", "Configured league is absent from catalogue")
            # Operational default is the current season; explicitly request all seasons for a backfill.
            available_seasons = list(record.get("seasons", []))
            seasons = [max(available_seasons, key=lambda value: int(str(value).split("/")[0]))] if available_seasons else []
        match_frames = []
        raw_matches = {}
        for season in seasons:
            matches = client.league_matches(league_id, season, status=status)
            cache.store(f"schedules_{league_id}_{season.replace('/', '-')}_{status}", f"/leagues/{league_id}/matches?season={season}&status={status}", {"matches": matches}, now=now, preserve_observation=True)
            if not matches:
                continue
            match_frames.append(normalize_matches(matches, league_id=league_id, season=season, observed_at=now))
            raw_matches.update({str(item["id"]): item for item in matches})
        if not match_frames:
            atomic_json(health_path, run)
            return frames
        incoming = pd.concat(match_frames, ignore_index=True).drop_duplicates("match_id", keep="last")
        incoming["league_key"] = config.key
        mapped, audit = reconcile_fixtures(incoming, canonical, aliases=config.team_aliases, mapped_at=now, max_hours=6.0)
        write_frame(audit, data_dir / "pitchapi_fixture_audit.csv")
        existing_mapping = read_frame(data_dir / "provider_fixture_map.csv")
        mapping_index = pd.concat([existing_mapping, mapped], ignore_index=True).drop_duplicates("provider_match_id", keep="last") if not existing_mapping.empty else mapped
        write_frame(mapping_index, data_dir / "provider_fixture_map.csv")
        unmatched = audit.loc[audit.status != "mapped"]
        for item in unmatched.itertuples(index=False):
            failure(f"FIXTURE_{item.status.upper()}", match_id=item.provider_match_id)
        incoming = incoming.merge(mapped[["provider_match_id", "fixture_id", "home_team_id", "away_team_id"]], left_on="match_id", right_on="provider_match_id", how="left", validate="one_to_one").drop(columns="provider_match_id")
        # Match metadata is a current index; raw response revisions remain separate.
        previous = frames["pitchapi_matches"]
        frames["pitchapi_matches"] = pd.concat([previous, incoming], ignore_index=True).drop_duplicates("match_id", keep="last").sort_values(["kickoff_utc", "match_id"])
        write_frame(frames["pitchapi_matches"], data_dir / "pitchapi_matches.csv")
        summary_rows = []
        for match in incoming.loc[incoming.fixture_id.notna()].itertuples(index=False):
            match_id, fixture_id = str(match.match_id), str(match.fixture_id)
            kickoff = utc_timestamp(match.kickoff_utc)
            team_ids = {str(match.home_team_id_provider): str(match.home_team_id), str(match.away_team_id_provider): str(match.away_team_id)}
            completed = match.status == "finished"
            completion = kickoff + timedelta(minutes=120) if completed else None
            fetched = {}
            requests_to_make = []
            if completed:
                requests_to_make.extend((name, endpoint) for enabled, name, endpoint in (
                    (with_shots, "shots", "shots"), (with_advanced, "advanced_team", "advanced"),
                    (with_players, "player_match", "advanced/players"), (with_momentum, "momentum", "momentum"),
                    (with_network, "network", "advanced/network"), (with_heatmaps, "heatmaps", "heatmaps"),
                ) if enabled)
            # Keep actual played lineups for analytics/history; eligibility is decided at feature time.
            if with_lineups:
                requests_to_make.append(("lineup_snapshots", "lineups"))
            for name, endpoint in requests_to_make:
                path = f"/matches/{match_id}/{endpoint}"
                try:
                    observation = cache.fetch(
                        f"{name}_{match_id}", path, lambda path=path: client._request(path), now=now,
                        completed_at=completion if name != "lineup_snapshots" else None,
                        maximum_age=timedelta(hours=1) if name == "lineup_snapshots" else timedelta(days=1),
                        refresh=refresh,
                        preserve_observation=name == "lineup_snapshots",
                    )
                    if observation.error_code:
                        failure(observation.error_code, match_id=match_id, endpoint=endpoint)
                    lineage = {"fixture_id": fixture_id, "match_id": match_id, "team_ids": team_ids, "observed_at": observation.observed_at}
                    if name == "shots":
                        normalized = normalize_shots(observation.payload, **lineage)
                        fetched["shots"] = observation
                    elif name == "advanced_team":
                        normalized = normalize_team_data(observation.payload, **lineage, home_provider_id=str(match.home_team_id_provider))
                    elif name == "player_match":
                        normalized = normalize_player_data(observation.payload, **lineage)
                    elif name == "lineup_snapshots":
                        normalized = normalize_lineups(observation.payload, fixture_id=fixture_id, match_id=match_id, home_team_id=str(match.home_team_id), away_team_id=str(match.away_team_id), snapshot_at=observation.observed_at, kickoff_utc=kickoff)
                    elif name == "network":
                        normalized = normalize_network(observation.payload, **lineage)
                    elif name == "heatmaps":
                        normalized = normalize_heatmaps(observation.payload, **lineage)
                    else:
                        normalized = pd.DataFrame([{**point, "fixture_id": fixture_id, "match_id": match_id, "observed_at": observation.observed_at.isoformat(), "provider_schema_version": INTEGRATION_SCHEMA_VERSION} for point in observation.payload.get("points", [])])
                    artifact = f"pitchapi_{name}"
                    keys = {"shots": ["fixture_id", "shot_id", "observed_at"], "advanced_team": ["fixture_id", "team_id", "observed_at"], "player_match": ["fixture_id", "player_id", "observed_at"], "lineup_snapshots": ["snapshot_id", "player_id"], "momentum": ["fixture_id", "minute", "observed_at"]}.get(name)
                    if keys is None:
                        # Networks/heatmaps retain response-level revision identity and deterministic row indices.
                        normalized = normalized.assign(provider_row_id=range(len(normalized)))
                        keys = ["fixture_id", "observed_at", "provider_row_id"]
                    frames[artifact] = append_revisions(normalized, data_dir / f"{artifact}{ARTIFACTS[artifact]}", keys=keys)
                    run["capabilities"].setdefault(name, {"successes": 0, "failures": 0})["successes"] += 1
                except (PitchAPIError, ValueError, TypeError, KeyError, OSError) as exc:
                    failure(getattr(exc, "code", type(exc).__name__), match_id=match_id, endpoint=endpoint)
                    run["capabilities"].setdefault(name, {"successes": 0, "failures": 0})["failures"] += 1
                    if not optional and getattr(exc, "code", None) not in {"ANALYTICS_UNAVAILABLE", "RESOURCE_NOT_FOUND"}:
                        raise
            if "shots" in fetched:
                observation = fetched["shots"]
                normalized = normalize_shots(observation.payload, fixture_id=fixture_id, match_id=match_id, team_ids=team_ids, observed_at=observation.observed_at)
                fixture = pd.DataFrame([{"fixture_id": fixture_id, "home_team_id": match.home_team_id, "away_team_id": match.away_team_id, "shots_available": True, "shots_observed_at": observation.observed_at.isoformat()}])
                summary = build_match_shot_features(normalized, fixture)
                summary_rows.append(summary)
                # Compatibility CSV keeps name/date keys expected by older consumer xG paths.
                xg = summary[["fixture_id", "home_xg", "away_xg", "observed_at"]].assign(match_id=match_id, match_date=match.match_date, HomeTeam=config.team_aliases.get(match.home_team, match.home_team), AwayTeam=config.team_aliases.get(match.away_team, match.away_team))
                frames["pitchapi_match_xg"] = pd.concat([frames["pitchapi_match_xg"], xg], ignore_index=True).drop_duplicates("fixture_id", keep="last")
        if summary_rows:
            frames["pitchapi_match_shot_features"] = append_revisions(pd.concat(summary_rows, ignore_index=True), data_dir / "pitchapi_match_shot_features.csv", keys=["fixture_id", "observed_at"])
            write_frame(frames["pitchapi_match_xg"], data_dir / "pitchapi_match_xg.csv")
    except (PitchAPIError, ValueError, TypeError, KeyError, OSError) as exc:
        failure(getattr(exc, "code", type(exc).__name__))
        if not optional:
            raise
    finally:
        atomic_json(health_path, run)
    return frames
