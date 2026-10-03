"""Construct live football and PitchAPI state at the forecast's actual cutoff."""

from pathlib import Path
import numpy as np
import pandas as pd

from pitch_oracle_core.config import LeagueConfig
from pitch_oracle_core.fixtures.canonical import canonical_fixture_frame
from .ledger import build_team_events, add_prior_team_state, match_feature_snapshots
from .pitchapi_mart import attach_pitchapi_features


def build_forecast_inputs(historical: pd.DataFrame, upcoming: pd.DataFrame, *, config: LeagueConfig, as_of: pd.Timestamp, data_dir: str | Path | None = None) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("Forecast input cutoff must be timezone-aware")
    if upcoming.empty:
        return upcoming.copy()
    history = canonical_fixture_frame(historical, config)
    targets = canonical_fixture_frame(upcoming, config, input_timezone=config.sources.upcoming_timezone).assign(as_of=cutoff)
    if data_dir is not None:
        from pitch_oracle_core.fixtures.registry import assign_fixture_ids
        registry = Path(data_dir) / "canonical_fixture_registry.json"
        history = assign_fixture_ids(history, registry, league_key=config.key, persist=False)
        targets = assign_fixture_ids(targets, registry, league_key=config.key, persist=False)
    if (targets.kickoff_lower_bound_utc <= cutoff).any():
        raise ValueError("Upcoming forecast inputs must predate known kickoff lower bounds")
    required = {"FullTimeHomeGoals", "FullTimeAwayGoals", "FullTimeResult"}
    if required.difference(history):
        raise ValueError("Forecast history requires completed result columns")
    prior = history.loc[history.FullTimeResult.isin(("H", "D", "A")) & (history.kickoff_utc + pd.Timedelta(hours=2) <= cutoff)]
    rename = {"FullTimeHomeGoals": "home_goals", "FullTimeAwayGoals": "away_goals", "HomeShots": "home_shots", "AwayShots": "away_shots", "HomeShotsOnTarget": "home_shots_on_target", "AwayShotsOnTarget": "away_shots_on_target"}
    output = []
    for _, target in targets.iterrows():
        # Include just this target, so earlier scheduled matches cannot alter the
        # rolling window for a forecast issued before either match has happened.
        synthetic = target.to_dict()
        synthetic.update(FullTimeHomeGoals=np.nan, FullTimeAwayGoals=np.nan)
        sequence = pd.concat([prior.loc[prior.fixture_id != target.fixture_id], pd.DataFrame([synthetic])], ignore_index=True).rename(columns=rename)
        events = build_team_events(sequence.drop(columns=["home_xg", "away_xg"], errors="ignore"))
        state = match_feature_snapshots(add_prior_team_state(events))
        state = state.loc[state.fixture_id == target.fixture_id].iloc[0]
        row = target.to_dict()
        row.update(state.to_dict())
        for canonical, legacy in {"home_points_l5": "HomeTeamPointsLast5", "away_points_l5": "AwayTeamPointsLast5", "home_rest_days": "HomeRestDays", "away_rest_days": "AwayRestDays", "home_goals_for_l5": "HomeGoalsAve", "away_goals_for_l5": "AwayGoalsAve"}.items():
            row[legacy] = row.get(canonical)
        output.append(row)
    targets = pd.DataFrame(output)
    sources = pd.concat([history, targets], ignore_index=True).drop_duplicates("fixture_id", keep="first")
    return attach_pitchapi_features(targets, sources, data_dir=data_dir, as_of=cutoff)
