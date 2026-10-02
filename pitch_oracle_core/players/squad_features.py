"""The same past-only player and lineup bridge for replay and live forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitch_oracle_core.features.families import TEAM_COLUMNS, LINEUP_COLUMNS, CONFIRMED_COLUMNS, KEEPER_COLUMNS
from pitch_oracle_core.pitchapi.normalize import boolean
from .goalkeeper import keeper_state
from .lineup_snapshots import latest_eligible_lineup, lineup_continuity, lineup_status
from .lineup_strength import LineupMember, PlayerStrength, lineup_delta, shrink_player
from .strength_estimator import player_strength_frame


def _starters(lineup: pd.DataFrame) -> set[str]:
    return set(lineup.loc[lineup.is_starter.map(boolean).eq(True), "player_id"].astype(str)) if not lineup.empty else set()


def _reference_roster(history: pd.DataFrame, team_id: str) -> list[str]:
    team = history.loc[history.team_id.astype(str).eq(str(team_id)) & pd.to_numeric(history.minutes, errors="coerce").gt(0)]
    if team.empty:
        return []
    recent_fixtures = team.sort_values("source_kickoff_utc").fixture_id.drop_duplicates().tail(5)
    minutes = team.loc[team.fixture_id.isin(recent_fixtures)].groupby("player_id").minutes.sum().sort_values(ascending=False, kind="stable")
    return minutes.head(11).index.astype(str).tolist()


def build_squad_features(targets: pd.DataFrame, players: pd.DataFrame, snapshots: pd.DataFrame, source_fixtures: pd.DataFrame, *, keeper_matches: pd.DataFrame | None = None, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    numeric = tuple(f"{side}_{name}" for side in ("home", "away") for name in (*TEAM_COLUMNS, *LINEUP_COLUMNS, *CONFIRMED_COLUMNS, *KEEPER_COLUMNS))
    source_fixtures = source_fixtures.copy()
    source_fixtures["kickoff_utc"] = pd.to_datetime(source_fixtures.kickoff_utc, utc=True, errors="raise")
    history = players.copy()
    if not history.empty:
        if source_fixtures.fixture_id.duplicated().any():
            raise ValueError("Player source fixture identity must be unique")
        source = source_fixtures[["fixture_id", "kickoff_utc", "league_key"]].copy()
        source["source_kickoff_lower_bound"] = source_fixtures.get("kickoff_lower_bound_utc", source_fixtures.kickoff_utc)
        history = history.drop(columns=["source_kickoff_utc"], errors="ignore").merge(source.rename(columns={"kickoff_utc": "source_kickoff_utc", "league_key": "source_league_key"}), on="fixture_id", how="left", validate="many_to_one")
        history["league_key"] = history.source_league_key
        for column in ("observed_at", "source_kickoff_utc"):
            history[column] = pd.to_datetime(history[column], utc=True, errors="raise")
        if history[["observed_at", "source_kickoff_utc"]].isna().any().any() or (history.observed_at <= pd.to_datetime(history.source_kickoff_lower_bound, utc=True)).any():
            raise ValueError("Player observations require mapped post-match lineage")
        history["minutes"] = pd.to_numeric(history.minutes, errors="coerce")
    output = []
    for target in targets.itertuples(index=False):
        kickoff = pd.Timestamp(target.kickoff_utc)
        default_cutoff = pd.Timestamp(getattr(target, "kickoff_lower_bound_utc", kickoff)) - pd.Timedelta(microseconds=1)
        cutoff = pd.Timestamp(as_of if as_of is not None else getattr(target, "as_of", default_cutoff))
        if cutoff.tzinfo is None or kickoff.tzinfo is None or cutoff >= pd.Timestamp(getattr(target, "kickoff_lower_bound_utc", kickoff)):
            raise ValueError("Squad features require a pre-kickoff cutoff")
        row = {"fixture_id": target.fixture_id, **{name: np.nan for name in numeric}}
        eligible = history.loc[(history.fixture_id != target.fixture_id) & (history.source_kickoff_utc < cutoff) & (history.observed_at <= cutoff)].sort_values("observed_at").drop_duplicates(["fixture_id", "player_id"], keep="last") if not history.empty else history
        strength_frame = player_strength_frame(eligible, estimated_at=cutoff.to_pydatetime())
        known = strength_frame.loc[strength_frame.metric_coverage > 0]
        strengths = {str(p.player_id): PlayerStrength(str(p.player_id), p.attack_per_90, p.defense_per_90, p.effective_minutes, cutoff.to_pydatetime(), p.model_id, p.prior_minutes) for p in known.itertuples(index=False)}
        keeper_history = keeper_matches if keeper_matches is not None else pd.DataFrame()
        if not keeper_history.empty:
            keeper_history = keeper_history.loc[(keeper_history.fixture_id != target.fixture_id) & (pd.to_datetime(keeper_history.source_kickoff_utc, utc=True) < cutoff)]
        keeper_states = keeper_state(keeper_history, target_kickoff=kickoff, as_of=cutoff)
        lineage = list(pd.to_datetime(known.source_observed_at, utc=True)) if not known.empty else []
        for side in ("home", "away"):
            team_id = str(getattr(target, f"{side}_team_id"))
            reference_ids = _reference_roster(eligible, team_id) if not eligible.empty else []
            reference = [LineupMember(player, 90, 1) for player in reference_ids]
            base_attack, base_defense, base_coverage = lineup_delta(reference, strengths, 0, 0)
            row[f"{side}_team_player_attack"] = base_attack if base_coverage else np.nan
            row[f"{side}_team_player_defense"] = base_defense if base_coverage else np.nan
            row[f"{side}_team_player_coverage"] = base_coverage
            lineup = latest_eligible_lineup(snapshots, fixture_id=target.fixture_id, team_id=team_id, as_of=cutoff, require_complete=True)
            status = lineup_status(lineup)
            row[f"{side}_lineup_status"] = status
            row[f"{side}_lineup_observed_at"] = pd.to_datetime(lineup.snapshot_at, utc=True).max().isoformat() if not lineup.empty else None
            row[f"{side}_lineup_is_confirmed"] = float(status == "confirmed")
            row[f"{side}_lineup_coverage"] = 0.0
            keeper_id = None
            if not lineup.empty:
                lineage.extend(pd.to_datetime(lineup.snapshot_at, utc=True).tolist())
                starters = _starters(lineup)
                attack, defense, coverage = lineup_delta([LineupMember(player, 90, 1) for player in sorted(starters)], strengths, 0, 0)
                row[f"{side}_lineup_coverage"] = coverage
                if coverage > 0 and base_coverage > 0:
                    row[f"{side}_lineup_attack_delta"] = attack - base_attack
                    row[f"{side}_lineup_defense_delta"] = defense - base_defense
                    missing = set(reference_ids).difference(starters)
                    row[f"{side}_lineup_missing_starter_value"] = sum(shrink_player(strengths[player].attack_per_90 + strengths[player].defense_per_90, strengths[player].effective_minutes, strengths[player].prior_minutes) for player in missing if player in strengths)
                    if status == "confirmed":
                        row[f"{side}_confirmed_lineup_attack_delta"] = attack - base_attack
                        row[f"{side}_confirmed_lineup_defense_delta"] = defense - base_defense
                        row[f"{side}_confirmed_lineup_coverage"] = coverage
                bench_ids = set(lineup.loc[lineup.is_starter.map(boolean).eq(False), "player_id"].astype(str))
                for metric in ("attack", "defense"):
                    values = [shrink_player(getattr(strengths[player], f"{metric}_per_90"), strengths[player].effective_minutes, strengths[player].prior_minutes) for player in sorted(bench_ids) if player in strengths]
                    row[f"{side}_lineup_bench_{metric}"] = float(np.mean(values)) if values else np.nan
                prior_fixtures = source_fixtures.loc[(source_fixtures.kickoff_utc < cutoff) & ((source_fixtures.home_team_id.astype(str) == team_id) | (source_fixtures.away_team_id.astype(str) == team_id))].sort_values("kickoff_utc", ascending=False)
                for prior in prior_fixtures.itertuples(index=False):
                    previous = latest_eligible_lineup(snapshots, fixture_id=prior.fixture_id, team_id=team_id, as_of=cutoff, require_complete=True)
                    if not previous.empty:
                        row[f"{side}_lineup_continuity"] = lineup_continuity(starters, _starters(previous))
                        break
                # Formation slots are never used to invent goalkeeper identities.
                historical_roles = strength_frame.set_index("player_id").role.to_dict() if not strength_frame.empty else {}
                keeper_ids = [str(p.player_id) for p in lineup.itertuples(index=False) if boolean(p.is_starter) is True and (str(p.position).upper() == "GK" or historical_roles.get(str(p.player_id)) == "GK")]
                if len(keeper_ids) == 1:
                    keeper_id = keeper_ids[0]
                    row[f"{side}_keeper_start_probability"] = 1.0 if status == "confirmed" else np.nan
            row[f"{side}_keeper_id"] = keeper_id
            state = keeper_states.loc[keeper_states.player_id == keeper_id] if keeper_id is not None else pd.DataFrame()
            if not state.empty:
                keeper = state.iloc[0]
                for name in KEEPER_COLUMNS:
                    source = name.removeprefix("keeper_") if name in {"keeper_effective_minutes", "keeper_shots_on_target_faced"} else name
                    if source in keeper:
                        row[f"{side}_{name}"] = keeper[source]
                lineage.append(pd.Timestamp(keeper.source_observed_at))
        row["squad_feature_observed_at"] = max(lineage).isoformat() if lineage else None
        output.append(row)
    return pd.DataFrame(output, columns=["fixture_id", *numeric, "home_lineup_status", "away_lineup_status", "home_lineup_observed_at", "away_lineup_observed_at", "home_keeper_id", "away_keeper_id", "squad_feature_observed_at"])
