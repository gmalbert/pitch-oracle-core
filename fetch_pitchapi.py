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
    league: LeagueConfig | str = "epl",
    *,
    seasons: list[str] | None = None,
    with_shots: bool = True,
    with_advanced: bool = False,
    with_momentum: bool = False,
    with_players: bool = False,
    with_lineups: bool = False,
    output_dir: str | Path | None = None,
    api_key: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch PitchAPI data for a league and write CSVs under ``output_dir``.

    Returns a dict of frame name -> DataFrame, and writes each to
    ``data_files/pitchapi_*.csv`` (tab-separated, matching repo convention).
    """
    config = get_league_config(league) if isinstance(league, str) else league
    league_id = pitchapi_league_id(config)
    key = api_key or os.getenv("PITCH_API_KEY")
    if not key:
        raise RuntimeError("PITCH_API_KEY is required to fetch PitchAPI data")
    active_output = Path(output_dir or os.getenv("PITCH_ORACLE_DATA_DIR", config.data_dir_name))
    active_output.mkdir(parents=True, exist_ok=True)
    client = PitchAPIClient(api_key=key)

    # Resolve the league seasons from the catalogue when not provided.
    if seasons is None:
        all_leagues = client.leagues()
        league_record = next((lg for lg in all_leagues if lg["id"] == league_id), None)
        if league_record is None:
            raise PitchAPIError(f"league {league_id} not found in PitchAPI catalogue")
        seasons = list(league_record.get("seasons", []))

    matches: list[dict] = []
    shots: list[dict] = []
    match_xg: list[dict] = []
    advanced: list[dict] = []
    momentum: list[dict] = []
    players: list[dict] = []
    lineups: list[dict] = []
    for season in seasons:
        season_matches = client.league_matches(league_id, season)
        print(f"  {season}: {len(season_matches)} matches")
        for match in season_matches:
            match_id = match["id"]
            matches.append({
                "match_id": match_id,
                "league_id": league_id,
                "date": match.get("date"),
                "time_utc": match.get("time_utc"),
                "status": match.get("status"),
                "home_team_id": match.get("home_team", {}).get("id"),
                "home_team": match.get("home_team", {}).get("name"),
                "away_team_id": match.get("away_team", {}).get("id"),
                "away_team": match.get("away_team", {}).get("name"),
                "score_home": match.get("score_home"),
                "score_away": match.get("score_away"),
                "round_name": match.get("round_name"),
            })
            if match.get("status") != "finished":
                continue
            if with_shots:
                periods = client.get_cached_or_fetch(
                    active_output, f"shots_{match_id}", f"/matches/{match_id}/shots"
                ).get("periods", [])
                shots.extend(_shot_rows(match, periods))
                match_xg.append(_match_xg_row(match, periods, config.team_aliases))
            if with_advanced:
                team_data = client.get_cached_or_fetch(
                    active_output, f"advanced_{match_id}", f"/matches/{match_id}/advanced"
                )
                if team_data:
                    for team in team_data.get("teams", []):
                        row = _flatten_team_advanced(team)
                        row["match_id"] = match_id
                        advanced.append(row)
            if with_momentum:
                points = client.get_cached_or_fetch(
                    active_output, f"momentum_{match_id}", f"/matches/{match_id}/momentum"
                ).get("points", [])
                for point in points:
                    momentum.append({"match_id": match_id, **point})
            if with_players:
                for line in client.get_cached_or_fetch(
                    active_output, f"players_{match_id}", f"/matches/{match_id}/players"
                ):
                    players.append({"match_id": match_id, **line})
            if with_lineups:
                lineups_data = client.get_cached_or_fetch(
                    active_output, f"lineups_{match_id}", f"/matches/{match_id}/lineups"
                )
                if lineups_data:
                    lineups.append({"match_id": match_id, **lineups_data})

    frames = {
        "pitchapi_matches": pd.DataFrame(matches),
        "pitchapi_shots": pd.DataFrame(shots),
        "pitchapi_match_xg": pd.DataFrame(match_xg),
        "pitchapi_advanced_team": pd.DataFrame(advanced),
        "pitchapi_momentum": pd.DataFrame(momentum),
        "pitchapi_players": pd.DataFrame(players),
        "pitchapi_lineups": pd.DataFrame(lineups),
    }
    for name, frame in frames.items():
        if frame.empty:
            continue
        frame.to_csv(active_output / f"{name}.csv", sep="\t", index=False)
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch PitchAPI football data.")
    parser.add_argument("--league", default=os.getenv("PITCH_ORACLE_LEAGUE", "epl"))
    parser.add_argument("--data-dir", default=os.getenv("PITCH_ORACLE_DATA_DIR", "data_files"))
    parser.add_argument("--seasons", nargs="*", default=None,
                        help="Season codes to fetch; defaults to all available.")
    parser.add_argument("--shots", action="store_true", default=True,
                        help="Fetch per-shot data and match-level xG (default).")
    parser.add_argument("--no-shots", dest="shots", action="store_false")
    parser.add_argument("--advanced", action="store_true")
    parser.add_argument("--momentum", action="store_true")
    parser.add_argument("--players", action="store_true")
    parser.add_argument("--lineups", action="store_true")
    args = parser.parse_args(argv)
    frames = fetch_pitchapi(
        args.league,
        seasons=args.seasons,
        with_shots=args.shots,
        with_advanced=args.advanced,
        with_momentum=args.momentum,
        with_players=args.players,
        with_lineups=args.lineups,
        output_dir=args.data_dir,
    )
    for name, frame in frames.items():
        if not frame.empty:
            print(f"Wrote {len(frame)} rows to {name}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
