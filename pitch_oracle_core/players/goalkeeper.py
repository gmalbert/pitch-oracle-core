"""Goalkeeper values with exposure, non-own goals, and strong sample shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .lineup_strength import shrink_player
from pitch_oracle_core.pitchapi.normalize import boolean


def build_keeper_matches(players: pd.DataFrame, shots: pd.DataFrame, fixtures: pd.DataFrame, *, response_revisions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join immutable player/shot revisions without inventing keeper attribution.

    Penalties in regulation are included; shootouts and own goals are excluded.
    Without substitution event timing, a match with multiple keepers or a keeper
    playing fewer than 89 minutes cannot safely attribute all opposing shots.
    Such a match contributes no keeper shot-stopping observation.
    """
    columns = ["fixture_id", "player_id", "team_id", "source_kickoff_utc", "observed_at", "minutes", "xgot_faced", "goals_conceded_non_own_goal", "shots_on_target_faced"]
    revisions = response_revisions if response_revisions is not None else pd.DataFrame()
    if players.empty or (shots.empty and revisions.empty):
        return pd.DataFrame(columns=columns)
    players, shots = players.copy(), shots.copy()
    if shots.empty:
        shots = pd.DataFrame(columns=["fixture_id", "observed_at", "opponent_id", "period", "is_own_goal", "is_on_target", "is_goal", "expected_goals_on_target"])
    for frame in (players, shots):
        frame["observed_at"] = pd.to_datetime(frame.observed_at, utc=True, errors="raise")
        if frame.observed_at.isna().any():
            raise ValueError("Keeper derivation requires actual observations")
    if fixtures.fixture_id.duplicated().any():
        raise ValueError("Keeper source fixtures must be unique")
    fixture_lookup = fixtures.set_index("fixture_id")
    output = []
    for fixture_id, player_history in players.groupby("fixture_id", sort=True):
        if fixture_id not in fixture_lookup.index:
            raise ValueError("Unmapped keeper source fixture")
        source = fixture_lookup.loc[fixture_id]
        kickoff = pd.Timestamp(source.kickoff_utc)
        shot_history = shots.loc[shots.fixture_id == fixture_id]
        shot_revisions = revisions.loc[revisions.fixture_id.eq(fixture_id) & revisions.artifact.eq("pitchapi_shots")].copy() if not revisions.empty else pd.DataFrame()
        if not shot_revisions.empty:
            shot_revisions["observed_at"] = pd.to_datetime(shot_revisions.observed_at, utc=True)
        if shot_history.empty and shot_revisions.empty:
            continue
        times = set(player_history.observed_at).union(shot_history.observed_at)
        times.update(shot_revisions.observed_at if not shot_revisions.empty else [])
        previous_keepers = {}
        for observed_at in sorted(times):
            if observed_at <= kickoff:
                raise ValueError("Keeper match observations must follow source kickoff")
            p = player_history.loc[player_history.observed_at <= observed_at].sort_values("observed_at").drop_duplicates("player_id", keep="last")
            if "is_deleted" in p:
                p = p.loc[~p.is_deleted.fillna(False).astype(bool)]
            s = shot_history.loc[shot_history.observed_at <= observed_at]
            clocks = shot_revisions.loc[shot_revisions.observed_at <= observed_at] if not shot_revisions.empty else pd.DataFrame()
            if s.empty and clocks.empty:
                continue
            # Select a complete shot response, so a removed shot does not survive a correction.
            clock = max(s.observed_at.max(), clocks.observed_at.max()) if not s.empty and not clocks.empty else clocks.observed_at.max() if not clocks.empty else s.observed_at.max()
            s = s.loc[s.observed_at == clock]
            current_keepers = {}
            for team_id in (source.home_team_id, source.away_team_id):
                keepers = p.loc[p.team_id.eq(team_id) & p.position.fillna("").str.upper().eq("GK") & pd.to_numeric(p.minutes, errors="coerce").gt(0)]
                if len(keepers) != 1 or float(keepers.iloc[0].minutes) < 89:
                    continue
                keeper = keepers.iloc[0]
                current_keepers[str(keeper.player_id)] = team_id
                faced = s.loc[s.opponent_id.eq(team_id) & ~s.period.fillna("").str.casefold().isin({"penaltyshootout", "penalty_shootout", "shootout"})]
                # Unknown own-goal/target flags cannot be assumed false.
                own_known = faced.is_own_goal.map(boolean).notna().all()
                ordinary = faced.loc[faced.is_own_goal.map(boolean).eq(False)]
                targets = ordinary.loc[ordinary.is_on_target.map(boolean).eq(True)]
                complete = own_known and ordinary.is_on_target.map(boolean).notna().all() and targets.expected_goals_on_target.notna().all()
                metrics = {"xgot_faced": float(targets.expected_goals_on_target.sum()) if complete else np.nan, "goals_conceded_non_own_goal": float(ordinary.is_goal.map(boolean).sum()) if complete and ordinary.is_goal.map(boolean).notna().all() else np.nan, "shots_on_target_faced": float(len(targets)) if complete else np.nan}
                row = {"fixture_id": fixture_id, "player_id": keeper.player_id, "team_id": team_id, "source_kickoff_utc": kickoff.isoformat(), "observed_at": observed_at.isoformat(), "minutes": float(keeper.minutes), **metrics}
                for name in ("keeper_claims", "keeper_claims_won", "keeper_claim_rate", "keeper_distributions", "keeper_distribution_accuracy", "keeper_sweeper_actions", "vaep_defensive"):
                    row[name] = keeper.get(name, np.nan)
                if pd.notna(row["keeper_distribution_accuracy"]) and pd.notna(row["keeper_distributions"]):
                    row["keeper_distributions_completed"] = row["keeper_distributions"] * row["keeper_distribution_accuracy"] / 100.0
                output.append(row)
            # A later correction that removes the keeper or invalidates attribution
            # must also invalidate the previously derived sample at later cutoffs.
            for player_id, team_id in previous_keepers.items():
                if player_id not in current_keepers:
                    output.append({"fixture_id": fixture_id, "player_id": player_id, "team_id": team_id, "source_kickoff_utc": kickoff.isoformat(), "observed_at": observed_at.isoformat(), "minutes": 0.0, "xgot_faced": np.nan, "goals_conceded_non_own_goal": np.nan, "shots_on_target_faced": np.nan})
            previous_keepers = current_keepers
    return pd.DataFrame(output) if output else pd.DataFrame(columns=columns)


def keeper_match_value(frame: pd.DataFrame) -> pd.Series:
    xgot = pd.to_numeric(frame["xgot_faced"], errors="coerce")
    goals = pd.to_numeric(frame["goals_conceded_non_own_goal"], errors="coerce")
    return xgot - goals


def keeper_state(matches: pd.DataFrame, *, target_kickoff: pd.Timestamp, as_of: pd.Timestamp | None = None, span: int = 10, prior_minutes: float = 1350.0) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of if as_of is not None else target_kickoff - pd.Timedelta(microseconds=1))
    if cutoff.tzinfo is None or pd.Timestamp(target_kickoff).tzinfo is None or cutoff >= target_kickoff:
        raise ValueError("Keeper state requires a pre-kickoff timezone-aware cutoff")
    if span < 1 or prior_minutes <= 0:
        raise ValueError("Keeper span and prior must be positive")
    if matches.empty:
        return pd.DataFrame(columns=["player_id", "keeper_strength", "effective_minutes", "shots_on_target_faced", "sample_status", "source_observed_at"])
    required = {"fixture_id", "player_id", "observed_at", "minutes", "xgot_faced", "goals_conceded_non_own_goal", "shots_on_target_faced"}
    if required.difference(matches):
        raise ValueError(f"Keeper history misses {sorted(required.difference(matches))}")
    frame = matches.copy()
    frame["observed_at"] = pd.to_datetime(frame.observed_at, utc=True, errors="raise")
    if frame.observed_at.isna().any():
        raise ValueError("Keeper history requires observation lineage")
    frame = frame.loc[frame.observed_at <= cutoff].sort_values("observed_at").drop_duplicates(["fixture_id", "player_id"], keep="last")
    if "source_kickoff_utc" in frame:
        frame["source_kickoff_utc"] = pd.to_datetime(frame.source_kickoff_utc, utc=True, errors="raise")
        if frame.source_kickoff_utc.isna().any() or (frame.source_kickoff_utc >= cutoff).any():
            raise ValueError("Keeper source fixture must predate the forecast cutoff")
        frame = frame.sort_values(["source_kickoff_utc", "fixture_id"])
    frame["xgot_prevented"] = keeper_match_value(frame)
    rows = []
    for player_id, history in frame.groupby("player_id", sort=True):
        history = history.tail(span)
        minutes = pd.to_numeric(history.minutes, errors="coerce")
        shots = pd.to_numeric(history.shots_on_target_faced, errors="coerce")
        if (minutes.dropna() < 0).any() or (shots.dropna() < 0).any():
            raise ValueError("Keeper exposure cannot be negative")
        valid = minutes.gt(0) & history.xgot_prevented.notna() & shots.notna()
        sample_minutes = float(minutes.loc[valid].sum())
        exposure = float(shots.loc[valid].sum())
        raw_rate = float(history.loc[valid, "xgot_prevented"].sum()) * 90 / sample_minutes if sample_minutes else 0.0
        sample = "strong_prior" if sample_minutes < 450 else "moderate_prior" if sample_minutes < 1350 else "normal"
        row = {"player_id": player_id, "keeper_strength": shrink_player(raw_rate, sample_minutes, prior_minutes), "effective_minutes": sample_minutes, "shots_on_target_faced": exposure, "sample_status": sample, "source_observed_at": history.observed_at.max().isoformat(), "keeper_xgot_prevented_per_shot": float(history.loc[valid, "xgot_prevented"].sum()) / exposure if exposure > 0 else np.nan}
        for numerator, denominator, label in (("keeper_claims_won", "keeper_claims", "keeper_claim_rate"), ("keeper_distributions_completed", "keeper_distributions", "keeper_distribution_accuracy")):
            if numerator in history and denominator in history:
                a, b = pd.to_numeric(history[numerator], errors="coerce"), pd.to_numeric(history[denominator], errors="coerce")
                known = a.notna() & b.gt(0)
                row[label] = float(a.loc[known].sum() / b.loc[known].sum()) if known.any() else np.nan
        for source, label in (("vaep_defensive", "keeper_vaep_defensive"), ("keeper_sweeper_actions", "keeper_sweeper_actions")):
            values = pd.to_numeric(history.get(source, pd.Series(np.nan, index=history.index)), errors="coerce")
            known = values.notna() & minutes.gt(0)
            row[label] = float(values.loc[known].sum() * 90 / minutes.loc[known].sum()) if known.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
