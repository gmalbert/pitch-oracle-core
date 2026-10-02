"""Fetch PitchAPI football data: match results, per-shot xG, and advanced analytics.

PitchAPI is a read-only REST API (https://pitchapi.dev). All endpoints are
historical matches and upcoming fixtures. The package client separates pre-match
observations from post-match analytics and preserves immutable raw revisions.

Auth: ``PITCH_API_KEY`` env var, sent as ``X-API-KEY`` on every request.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.leagues import get_league_config
from pitch_oracle_core.team_mappings import normalize_team_name


BASE_URL = "https://api.pitchapi.dev/v1"
DEFAULT_CACHE_DIR = "pitchapi_cache"
MATCH_EPOCH_SECONDS = 30 * 60  # per-match payloads are immutable once finished


from pitch_oracle_core.pitchapi.client import PitchAPIClient, PitchAPIError


def pitchapi_league_id(config: LeagueConfig) -> str:
    league_id = config.sources.pitchapi_league_id
    if not config.sources.pitchapi or not league_id:
        raise ValueError(f"PitchAPI is not configured for {config.key}")
    return league_id


def _shot_rows(match: dict, periods: list[dict]) -> list[dict]:
    rows = []
    for period in periods:
        for shot in period.get("shots", []):
            player = shot.get("player") or {}
            rows.append({
                "match_id": match.get("id"),
                "match_date": match.get("date"),
                "team_id": shot.get("team_id"),
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "minute": shot.get("minute"),
                "minute_added": shot.get("minute_added"),
                "x": shot.get("x"),
                "y": shot.get("y"),
                "expected_goals": shot.get("expected_goals"),
                "expected_goals_on_target": shot.get("expected_goals_on_target"),
                "is_on_target": shot.get("is_on_target"),
                "goal_crossed_y": shot.get("goal_crossed_y"),
                "goal_crossed_z": shot.get("goal_crossed_z"),
                "is_inside_box": shot.get("is_inside_box"),
                "shot_type": shot.get("shot_type"),
                "situation": shot.get("situation"),
                "event_type": shot.get("event_type"),
                "is_blocked": shot.get("is_blocked"),
                "is_own_goal": shot.get("is_own_goal"),
            })
    return rows


def _match_xg_row(match: dict, periods: list[dict], aliases: dict[str, str] | None = None) -> dict:
    home_team_id = match.get("home_team", {}).get("id")
    away_team_id = match.get("away_team", {}).get("id")
    total = {"home": 0.0, "away": 0.0}
    for period in periods:
        for shot in period.get("shots", []):
            team = shot.get("team_id", "")
            side = "home" if team == home_team_id else "away"
            total[side] += float(shot.get("expected_goals") or 0.0)
    aliases = aliases or {}
    return {
        "match_id": match.get("id"),
        "match_date": match.get("date"),
        "HomeTeam": normalize_team_name(match.get("home_team", {}).get("name", ""), aliases),
        "AwayTeam": normalize_team_name(match.get("away_team", {}).get("name", ""), aliases),
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_xg": round(total["home"], 4),
        "away_xg": round(total["away"], 4),
    }


def _flatten_team_advanced(team: dict) -> dict:
    row = {"team_id": team.get("team", {}).get("id")}
    for group, value in team.items():
        if group in ("team",):
            continue
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, dict):
                    for subkey, subvalue in item.items():
                        row[f"{group}.{key}.{subkey}"] = subvalue
                else:
                    row[f"{group}.{key}"] = item
        else:
            row[group] = value
    return row


def fetch_pitchapi(
    league: LeagueConfig | str = "epl", *, seasons: list[str] | None = None,
    with_shots: bool = True, with_advanced: bool = False,
    with_momentum: bool = False, with_players: bool = False,
    with_lineups: bool = False, with_network: bool = False,
    with_heatmaps: bool = False, output_dir: str | Path | None = None,
    api_key: str | None = None, status: str = "all", optional: bool = True,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compatibility entry point for the versioned package ingestion pipeline."""
    from pitch_oracle_core.pitchapi.ingest import refresh_pitchapi
    return refresh_pitchapi(
        league, seasons=seasons, output_dir=output_dir, api_key=api_key,
        with_shots=with_shots, with_advanced=with_advanced,
        with_momentum=with_momentum, with_players=with_players,
        with_lineups=with_lineups, with_network=with_network,
        with_heatmaps=with_heatmaps, status=status, optional=optional,
        refresh=refresh,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch PitchAPI football data.")
    parser.add_argument("--league", default=os.getenv("PITCH_ORACLE_LEAGUE", "epl"))
    parser.add_argument("--data-dir", default=os.getenv("PITCH_ORACLE_DATA_DIR", "data_files"))
    parser.add_argument("--seasons", nargs="*", default=None,
                        help="Season codes to fetch; defaults to the latest available season.")
    parser.add_argument("--shots", action="store_true", default=True,
                        help="Fetch per-shot data and match-level xG (default).")
    parser.add_argument("--no-shots", dest="shots", action="store_false")
    parser.add_argument("--advanced", action="store_true")
    parser.add_argument("--momentum", action="store_true")
    parser.add_argument("--players", action="store_true")
    parser.add_argument("--lineups", action="store_true")
    parser.add_argument("--network", action="store_true")
    parser.add_argument("--heatmaps", action="store_true")
    parser.add_argument("--status", choices=["played", "upcoming", "all"], default="all")
    parser.add_argument("--strict", action="store_true", help="Fail on transport/schema failures instead of retaining optional cache")
    parser.add_argument("--refresh", action="store_true", help="Explicitly recheck completed payloads outside the correction window")
    args = parser.parse_args(argv)
    frames = fetch_pitchapi(
        args.league,
        seasons=args.seasons,
        with_shots=args.shots,
        with_advanced=args.advanced,
        with_momentum=args.momentum,
        with_players=args.players,
        with_lineups=args.lineups,
        with_network=args.network, with_heatmaps=args.heatmaps,
        status=args.status, optional=not args.strict, refresh=args.refresh,
        output_dir=args.data_dir,
    )
    for name, frame in frames.items():
        if not frame.empty:
            print(f"Wrote {len(frame)} rows to {name}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
