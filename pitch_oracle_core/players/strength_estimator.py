"""Past-only minute-weighted player rates with recency, role scaling, and priors."""

from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd

from pitch_oracle_core.pitchapi.contracts import utc_timestamp
from .lineup_strength import PlayerStrength

ATTACK_WEIGHTS = {"vaep_offensive": .30, "xt_total": .20, "xag": .15, "xg_chain": .10, "progressive_passes": .10, "progressive_carries": .10, "gca": .05}
DEFENSE_WEIGHTS = {"vaep_defensive": .40, "tackles": .15, "interceptions": .15, "duels_won": .15, "aerials_won": .15}
ROLE_MAP = {"GK": "GK", "GOALKEEPER": "GK", "CB": "CB", "LB": "FB/WB", "RB": "FB/WB", "LWB": "FB/WB", "RWB": "FB/WB", "DM": "DM/CM", "CDM": "DM/CM", "CM": "DM/CM", "AM": "AM/W", "CAM": "AM/W", "LW": "AM/W", "RW": "AM/W", "ST": "ST", "CF": "ST", "DF": "DF", "MF": "MF", "FW": "FW"}


def player_strength_frame(
    history: pd.DataFrame, *, estimated_at: datetime, model_id: str = "pitchapi-player-strength-v1",
    half_life_days: float = 180.0, league_factors: dict[str, float] | None = None,
) -> pd.DataFrame:
    cutoff = utc_timestamp(estimated_at)
    if half_life_days <= 0:
        raise ValueError("Recency half-life must be positive")
    columns = ["player_id", "team_id", "role", "attack_per_90", "defense_per_90", "effective_minutes", "history_matches", "metric_coverage", "estimated_at", "source_observed_at", "model_id", "prior_minutes"]
    if history.empty:
        return pd.DataFrame(columns=columns)
    required = {"fixture_id", "player_id", "team_id", "observed_at", "minutes"}
    if required.difference(history):
        raise ValueError(f"Player history misses {sorted(required.difference(history))}")
    frame = history.copy()
    frame["observed_at"] = pd.to_datetime(frame.observed_at, utc=True, errors="raise")
    if frame.observed_at.isna().any():
        raise ValueError("Player history requires actual observation times")
    frame = frame.loc[frame.observed_at <= cutoff].sort_values("observed_at", kind="stable").drop_duplicates(["fixture_id", "player_id"], keep="last")
    frame["minutes"] = pd.to_numeric(frame.minutes, errors="coerce")
    if (frame.minutes.dropna() < 0).any() or (frame.minutes.dropna() > 130).any():
        raise ValueError("Invalid player-match minutes")
    frame = frame.loc[frame.minutes > 0].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    # The observation timestamp governs availability; match kickoff governs recency when known.
    match_time = pd.to_datetime(frame.get("source_kickoff_utc", frame.observed_at), utc=True, errors="raise")
    if (match_time > cutoff).any():
        raise ValueError("Player source fixture is in the future")
    frame["_weight"] = np.exp2(-((pd.Timestamp(cutoff) - match_time).dt.total_seconds() / 86400) / half_life_days)
    metrics = tuple(dict.fromkeys([*ATTACK_WEIGHTS, *DEFENSE_WEIGHTS]))
    estimates = []
    for player_id, player in frame.groupby("player_id", sort=True):
        recent = player.iloc[-1]
        role = ROLE_MAP.get(str(recent.get("position", "")).upper(), "unknown")
        minutes = player.minutes * player._weight
        row = {"player_id": str(player_id), "team_id": str(recent.team_id), "role": role, "league_key": str(recent.get("league_key", "unknown")), "effective_minutes": float(minutes.sum()), "history_matches": player.fixture_id.nunique(), "estimated_at": cutoff.isoformat(), "source_observed_at": player.observed_at.max().isoformat(), "model_id": model_id, "prior_minutes": 1350.0 if role == "GK" else 900.0}
        for metric in metrics:
            raw = pd.to_numeric(player.get(metric, pd.Series(np.nan, index=player.index)), errors="coerce")
            known = raw.notna()
            exposure = float(minutes.loc[known].sum())
            factors = player.get("league_key", pd.Series("unknown", index=player.index)).map(league_factors or {}).fillna(1.0)
            if (factors <= 0).any():
                raise ValueError("League adjustment factors must be positive")
            row[f"{metric}_per90"] = float((raw.loc[known] * player.loc[known, "_weight"] * factors.loc[known]).sum()) * 90 / exposure if exposure > 0 else np.nan
        row["metric_coverage"] = float(sum(pd.notna(row[f"{metric}_per90"]) for metric in metrics) / len(metrics))
        estimates.append(row)
    result = pd.DataFrame(estimates)
    # All scaler statistics are computed from eligible history at this cutoff only.
    for _, group in result.groupby(["league_key", "role"], sort=False):
        for metric in metrics:
            column = f"{metric}_per90"
            valid = group[column].notna()
            if not valid.any():
                continue
            values = group.loc[valid, column].astype(float)
            weights = group.loc[valid, "effective_minutes"].astype(float)
            mean = float(np.average(values, weights=weights))
            variance = float(np.average((values - mean) ** 2, weights=weights))
            scaled = ((values - mean) / np.sqrt(variance)).clip(-4, 4) if variance > 1e-12 else values * 0
            result.loc[values.index, f"z_{metric}"] = scaled
    for target, weights in (("attack_per_90", ATTACK_WEIGHTS), ("defense_per_90", DEFENSE_WEIGHTS)):
        numerator = pd.Series(0.0, index=result.index)
        denominator = pd.Series(0.0, index=result.index)
        for metric, weight in weights.items():
            values = result.get(f"z_{metric}", pd.Series(np.nan, index=result.index))
            numerator += values.fillna(0) * weight
            denominator += values.notna() * weight
        result[target] = numerator.div(denominator.where(denominator > 0)).fillna(0)
    return result[[*columns, "league_key"]]


def estimate_player_strengths(history: pd.DataFrame, *, estimated_at: datetime, model_id: str = "pitchapi-player-strength-v1", **kwargs) -> dict[str, PlayerStrength]:
    frame = player_strength_frame(history, estimated_at=estimated_at, model_id=model_id, **kwargs)
    return {str(row.player_id): PlayerStrength(str(row.player_id), float(row.attack_per_90), float(row.defense_per_90), float(row.effective_minutes), utc_timestamp(row.estimated_at), str(row.model_id), float(row.prior_minutes)) for row in frame.itertuples(index=False)}
