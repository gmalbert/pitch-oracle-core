"""Explicit provider field whitelist; null remains unknown and IDs remain internal."""

from __future__ import annotations

from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd

from .contracts import INTEGRATION_SCHEMA_VERSION, utc_timestamp

GROUP_FIELDS = {
    "possession_value": "xt_total vaep_total vaep_offensive vaep_defensive pv_total pv_offensive pv_defensive",
    "passing": "passes pass_accuracy progressive_passes progressive_pass_distance passes_into_box key_passes assists crosses switches",
    "carrying": "carries progressive_carries carries_into_final_third carries_into_box take_ons_won miscontrols dispossessed",
    "creation": "sca gca xag xg_chain xg_buildup",
    "defending": "tackles interceptions duels_won aerials_won ppda avg_defensive_action_x high_turnovers counterpress_regains_5s ball_recovery_time",
    "territory": "field_tilt possession_pct final_third_entries box_entries",
    "tempo": "passes_per_sequence sequence_time direct_speed buildup_attacks direct_attacks",
    "goalkeeping": "claims claims_won claim_rate sweeper_actions distributions distribution_accuracy",
}
ADVANCED_FIELD_MAP = {(group, field): (f"keeper_{field}" if group == "goalkeeping" else field) for group, fields in GROUP_FIELDS.items() for field in fields.split()}
SHOT_COLUMNS = ["fixture_id", "match_id", "shot_id", "team_id", "opponent_id", "provider_team_id", "player_id", "provider_player_id", "player_name", "minute", "minute_added", "period", "x", "y", "expected_goals", "expected_goals_on_target", "is_on_target", "is_goal", "is_own_goal", "is_blocked", "is_inside_box", "body_part", "situation", "shot_type", "event_type", "goal_crossed_y", "goal_crossed_z", "observed_at", "provider_schema_version", "coordinate_frame"]
LINEUP_COLUMNS = ["fixture_id", "provider_match_id", "team_id", "player_id", "provider_player_id", "player_name", "position", "position_slot", "is_starter", "lineup_status", "snapshot_id", "snapshot_at", "kickoff_utc", "source", "formation", "lineup_type", "pitch_x", "pitch_y", "provider_schema_version"]


def number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def boolean(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    raise ValueError(f"Invalid provider boolean: {value!r}")


def get_nested(payload: dict, *keys: str):
    value = payload
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def normalize_advanced_team(team_payload: dict) -> dict:
    row = {output: number(get_nested(team_payload, *path)) for path, output in ADVANCED_FIELD_MAP.items()}
    team = team_payload.get("team") or {}
    row.update(provider_team_id=team.get("id"), provider_team_name=team.get("name"))
    return row


def _lineage(fixture_id: str, match_id: str, observed_at: datetime) -> dict:
    if not fixture_id or not match_id:
        raise ValueError("Canonical fixture and provider match identities are required")
    return {"fixture_id": fixture_id, "match_id": match_id, "observed_at": utc_timestamp(observed_at).isoformat(), "provider_schema_version": INTEGRATION_SCHEMA_VERSION}


def normalize_matches(matches: list[dict], *, league_id: str, season: str, observed_at: datetime) -> pd.DataFrame:
    rows = []
    for match in matches:
        kickoff = match.get("time_utc") or match.get("kickoff_utc")
        if kickoff is None:
            raise ValueError("Provider fixture lacks a kickoff timestamp")
        # Some legacy responses provide HH:MM separately; dates are explicitly UTC here.
        if isinstance(kickoff, str) and len(kickoff) in {5, 8}:
            kickoff = f"{match['date']}T{kickoff}Z"
        home, away = match.get("home_team") or {}, match.get("away_team") or {}
        rows.append({"match_id": str(match["id"]), "league_id": league_id, "season": season, "match_date": match.get("date"), "kickoff_utc": utc_timestamp(kickoff).isoformat(), "status": match.get("status"), "home_team_id_provider": home.get("id"), "away_team_id_provider": away.get("id"), "home_team_name_provider": home.get("name"), "away_team_name_provider": away.get("name"), "home_team": home.get("name"), "away_team": away.get("name"), "score_home": number(match.get("score_home")), "score_away": number(match.get("score_away")), "round_name": match.get("round_name"), "observed_at": utc_timestamp(observed_at).isoformat(), "provider_schema_version": INTEGRATION_SCHEMA_VERSION})
    return pd.DataFrame(rows)


def normalize_shots(payload: dict, *, fixture_id: str, match_id: str, team_ids: dict[str, str], observed_at: datetime) -> pd.DataFrame:
    if len(team_ids) != 2 or len(set(team_ids.values())) != 2:
        raise ValueError("Shot normalization requires two distinct canonical teams")
    rows = []
    for period in payload.get("periods", []):
        for shot in period.get("shots", []):
            provider_team = str(shot.get("team_id", ""))
            if provider_team not in team_ids:
                raise ValueError("Shot team does not belong to this fixture")
            x, y = number(shot.get("x")), number(shot.get("y"))
            if (x is not None and not 0 <= x <= 105) or (y is not None and not 0 <= y <= 68):
                raise ValueError("Shot coordinates are outside the 105 x 68 metre pitch")
            xg = number(shot.get("expected_goals"))
            xgot = number(shot.get("expected_goals_on_target"))
            if (xg is not None and xg < 0) or (xgot is not None and xgot < 0):
                raise ValueError("Expected goals cannot be negative")
            player = shot.get("player") or {}
            player_id = player.get("id")
            event = shot.get("event_type")
            row = {**_lineage(fixture_id, match_id, observed_at), "shot_id": shot.get("id"), "team_id": team_ids[provider_team], "opponent_id": next(value for key, value in team_ids.items() if key != provider_team), "provider_team_id": provider_team, "player_id": f"pitchapi:{player_id}" if player_id else None, "provider_player_id": player_id, "player_name": player.get("name"), "minute": number(shot.get("minute")), "minute_added": number(shot.get("minute_added")), "period": period.get("period"), "x": x, "y": y, "expected_goals": xg, "expected_goals_on_target": xgot, "is_goal": str(event).casefold() == "goal" if event else None, "body_part": shot.get("shot_type"), "coordinate_frame": "acting_ltr"}
            row.update({name: shot.get(name) for name in ("situation", "shot_type", "event_type")})
            row.update({name: number(shot.get(name)) for name in ("goal_crossed_y", "goal_crossed_z")})
            row.update({name: boolean(shot.get(name)) for name in ("is_on_target", "is_own_goal", "is_blocked", "is_inside_box")})
            rows.append(row)
    result = pd.DataFrame(rows, columns=SHOT_COLUMNS)
    if not result.empty and (result.shot_id.isna().any() or result.shot_id.duplicated().any()):
        raise ValueError("Every shot requires a unique ID scoped to its fixture")
    return result


def normalize_team_data(payload: dict, *, fixture_id: str, match_id: str, team_ids: dict[str, str], home_provider_id: str, observed_at: datetime) -> pd.DataFrame:
    rows = []
    for team in payload.get("teams", []):
        row = normalize_advanced_team(team)
        provider_id = str(row["provider_team_id"])
        if provider_id not in team_ids:
            raise ValueError("Advanced team does not belong to this fixture")
        rows.append({**row, **_lineage(fixture_id, match_id, observed_at), "team_id": team_ids[provider_id], "opponent_id": next(value for key, value in team_ids.items() if key != provider_id), "is_home": provider_id == home_provider_id})
    result = pd.DataFrame(rows)
    if not result.empty and result.team_id.duplicated().any():
        raise ValueError("Duplicate advanced fixture/team rows")
    return result


def normalize_player_data(payload: dict, *, fixture_id: str, match_id: str, team_ids: dict[str, str], observed_at: datetime) -> pd.DataFrame:
    rows = []
    for item in payload.get("players", []):
        provider_team = str(item.get("team_id", ""))
        if provider_team not in team_ids:
            raise ValueError("Player team does not belong to this fixture")
        player = item.get("player") or {}
        provider_player = player.get("id")
        if not provider_player:
            raise ValueError("Player identity is required")
        rows.append({**normalize_advanced_team(item), **_lineage(fixture_id, match_id, observed_at), "team_id": team_ids[provider_team], "provider_team_id": provider_team, "player_id": f"pitchapi:{provider_player}", "provider_player_id": provider_player, "player_name": player.get("name"), "minutes": number(item.get("minutes_played")), "position": "GK" if item.get("goalkeeping") is not None else item.get("position"), "rating": number(item.get("rating")), "starter": boolean(item.get("starter"))})
    result = pd.DataFrame(rows)
    if not result.empty and result.player_id.duplicated().any():
        raise ValueError("Duplicate player-match rows")
    return result


def normalize_lineups(payload: dict, *, fixture_id: str, match_id: str, home_team_id: str, away_team_id: str, snapshot_at: datetime, kickoff_utc: datetime) -> pd.DataFrame:
    snapshot_at, kickoff_utc = utc_timestamp(snapshot_at), utc_timestamp(kickoff_utc)
    rows = []
    for side, team_id in (("home", home_team_id), ("away", away_team_id)):
        lineup = payload.get(side)
        if not isinstance(lineup, dict):
            continue
        confirmed = boolean(lineup.get("confirmed"))
        status = "confirmed" if confirmed is True else "predicted" if confirmed is False else "unknown"
        snapshot_id = f"{match_id}:{side}:{snapshot_at.isoformat()}"
        for field, starter in (("starters", True), ("bench", False)):
            for player in lineup.get(field, []):
                provider_id = player.get("player_id") or player.get("id")
                if not provider_id:
                    raise ValueError("Lineup member requires a provider player ID")
                slot = player.get("position_id")
                # Starter position_id is a formation slot, NOT a position category.
                position = player.get("position") or ({0: "GK", 1: "DF", 2: "MF", 3: "FW"}.get(slot) if not starter else None)
                rows.append({"fixture_id": fixture_id, "provider_match_id": match_id, "team_id": team_id, "player_id": f"pitchapi:{provider_id}", "provider_player_id": provider_id, "player_name": player.get("name"), "position": position, "position_slot": slot, "is_starter": starter, "lineup_status": status, "snapshot_id": snapshot_id, "snapshot_at": snapshot_at.isoformat(), "kickoff_utc": kickoff_utc.isoformat(), "source": "pitchapi", "formation": lineup.get("formation"), "lineup_type": lineup.get("lineup_type"), "pitch_x": number(player.get("pitch_x")), "pitch_y": number(player.get("pitch_y")), "provider_schema_version": INTEGRATION_SCHEMA_VERSION})
    result = pd.DataFrame(rows, columns=LINEUP_COLUMNS)
    if not result.empty and result.duplicated(["team_id", "player_id"]).any():
        raise ValueError("Duplicate lineup member in an observation")
    return result


def normalize_network(payload: dict, *, fixture_id: str, match_id: str, team_ids: dict[str, str], observed_at: datetime) -> pd.DataFrame:
    rows = []
    for network in payload.get("networks", []):
        provider_id = str((network.get("team") or {}).get("id", ""))
        if provider_id not in team_ids:
            raise ValueError("Passing network team does not belong to fixture")
        shared = {**_lineage(fixture_id, match_id, observed_at), "team_id": team_ids[provider_id], "centralization": number(network.get("centralization")), "window_until_seconds": number((network.get("window") or {}).get("until_seconds")), "coordinate_frame": "acting_ltr"}
        for node in network.get("nodes", []):
            player = node.get("player") or {}
            if not player.get("id"):
                raise ValueError("Passing network node requires player identity")
            rows.append({**shared, "kind": "node", "player_id": f"pitchapi:{player.get('id')}", "player_name": player.get("name"), **{key: number(node.get(key)) for key in ("avg_x", "avg_y", "passes", "passes_received", "degree", "strength", "betweenness", "clustering")}})
        for edge in network.get("edges", []):
            if not (edge.get("from") or {}).get("id") or not (edge.get("to") or {}).get("id"):
                raise ValueError("Passing network edge requires player identities")
            rows.append({**shared, "kind": "edge", "from_player_id": f"pitchapi:{(edge.get('from') or {}).get('id')}", "to_player_id": f"pitchapi:{(edge.get('to') or {}).get('id')}", "passes": number(edge.get("passes"))})
    return pd.DataFrame(rows)


def normalize_heatmaps(payload: dict, *, fixture_id: str, match_id: str, team_ids: dict[str, str], observed_at: datetime) -> pd.DataFrame:
    grid = payload.get("grid") or {}
    length, width = int(grid.get("length", 16)), int(grid.get("width", 12))
    frame = grid.get("frame")
    if frame not in {"acting_ltr", "home_ltr"} or length <= 0 or width <= 0:
        raise ValueError("Heatmap grid requires a known coordinate frame and dimensions")
    rows = []
    for kind in ("teams", "players"):
        for item in payload.get(kind, []):
            provider_team = str((item.get("team") or {}).get("id", ""))
            if provider_team not in team_ids:
                raise ValueError("Heatmap team does not belong to fixture")
            player = item.get("player") or {}
            for cell_x, cell_y, actions in item.get("cells", []):
                if not 0 <= cell_x < length or not 0 <= cell_y < width or actions < 0:
                    raise ValueError("Heatmap cell outside its declared grid")
                rows.append({**_lineage(fixture_id, match_id, observed_at), "kind": kind[:-1], "team_id": team_ids[provider_team], "player_id": f"pitchapi:{player['id']}" if player.get("id") else None, "player_name": player.get("name"), "side": item.get("side"), "cell_x": cell_x, "cell_y": cell_y, "actions": actions, "grid_length": length, "grid_width": width, "coordinate_frame": frame})
    return pd.DataFrame(rows)
