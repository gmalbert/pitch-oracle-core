"""Observation-aware historical states for both forecasts and strict replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LedgerMetric:
    name: str
    column: str
    perspective: Literal["for_against", "team_only"] = "team_only"
    family: str = "advanced_team"
    spans: tuple[int, ...] = (5, 10)


METRICS = (
    LedgerMetric("xt", "xt_total", "for_against"),
    *(LedgerMetric(name, name) for name in (
        "vaep_total", "vaep_offensive", "vaep_defensive", "ppda", "field_tilt",
        "high_turnovers", "counterpress_regains_5s", "avg_defensive_action_x",
        "ball_recovery_time", "box_entries", "progressive_passes", "progressive_carries",
        "direct_speed", "passes_per_sequence",
    )),
    LedgerMetric("xg", "xg", "for_against", "xg"),
    LedgerMetric("finishing_vs_expectation", "finishing_vs_expectation", family="xg"),
    LedgerMetric("xgot", "xgot", "for_against", "shot_profile"),
    *(LedgerMetric(name, name, family="shot_profile") for name in (
        "shots", "shots_on_target", "xg_per_shot", "xgot_minus_xg", "inside_box_share",
        "set_piece_xg_share", "open_play_xg_share", "blocked_rate", "header_share",
        "chance_concentration",
    )),
)


def _timestamp(value, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{field} must be a valid timezone-aware timestamp")
    return timestamp.tz_convert("UTC")


def _prepare_observations(observations: pd.DataFrame, source_fixtures: pd.DataFrame) -> pd.DataFrame:
    required = {"fixture_id", "team_id", "observed_at"}
    if required.difference(observations):
        raise ValueError(f"Observation history misses {sorted(required.difference(observations))}")
    if "fixture_id" not in source_fixtures or "kickoff_utc" not in source_fixtures:
        raise ValueError("Source fixtures require canonical IDs and kickoff")
    if source_fixtures.fixture_id.duplicated().any():
        raise ValueError("Source fixture identity must be unique")
    source = source_fixtures[["fixture_id", "kickoff_utc", "home_team_id", "away_team_id"]].copy()
    source["source_kickoff_lower_bound"] = source_fixtures.get("kickoff_lower_bound_utc", source_fixtures.kickoff_utc)
    source = source.rename(columns={"kickoff_utc": "source_kickoff_utc"})
    frame = observations.drop(columns=["source_kickoff_utc"], errors="ignore").merge(source, on="fixture_id", how="left", validate="many_to_one")
    frame["observed_at"] = pd.to_datetime(frame.observed_at, utc=True, errors="raise")
    frame["source_kickoff_utc"] = pd.to_datetime(frame.source_kickoff_utc, utc=True, errors="raise")
    if frame[["observed_at", "source_kickoff_utc"]].isna().any().any():
        raise ValueError("Every observation needs actual lineage and a mapped source fixture")
    if (~((frame.team_id == frame.home_team_id) | (frame.team_id == frame.away_team_id))).any():
        raise ValueError("Observation team does not belong to its source fixture")
    if (frame.observed_at <= pd.to_datetime(frame.source_kickoff_lower_bound, utc=True)).any():
        raise ValueError("Post-match metrics cannot be observed before their source kickoff")
    if frame.duplicated(["fixture_id", "team_id", "observed_at"]).any():
        raise ValueError("Duplicate source fixture/team/observation revision")
    return frame


def shot_summaries_to_team_observations(summaries: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    if summaries.empty:
        return pd.DataFrame(columns=["fixture_id", "team_id", "observed_at"])
    rows = summaries.merge(fixtures[["fixture_id", "home_team_id", "away_team_id"]], on="fixture_id", validate="many_to_one")
    frames = []
    for side in ("home", "away"):
        metric_columns = [column for column in summaries if column.startswith(f"{side}_")]
        frame = rows[["fixture_id", "observed_at", f"{side}_team_id", *metric_columns]].copy()
        frame = frame.rename(columns={f"{side}_team_id": "team_id", **{column: column.removeprefix(f"{side}_") for column in metric_columns}})
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def feature_names(metrics: tuple[LedgerMetric, ...] = METRICS) -> tuple[str, ...]:
    names = []
    for metric in metrics:
        perspectives = ("for", "against") if metric.perspective == "for_against" else (None,)
        for side in ("home", "away"):
            for perspective in perspectives:
                stem = f"{side}_{metric.name}" + (f"_{perspective}" if perspective else "")
                names.extend([f"{stem}_previous", f"{stem}_l5", f"{stem}_l10", *(f"{stem}_ewm{span}" for span in metric.spans), f"{stem}_coverage_l10"])
    return tuple(names)


def build_advanced_team_features(
    targets: pd.DataFrame, observations: pd.DataFrame, source_fixtures: pd.DataFrame, *,
    as_of: pd.Timestamp | None = None, metrics: tuple[LedgerMetric, ...] = METRICS,
) -> pd.DataFrame:
    """Select actual observations known at each forecast cutoff, then calculate state.

    A previous fixture alone is insufficient: a late response or correction is
    excluded until its observation time. The target fixture is always excluded.
    Historical defaults evaluate immediately before kickoff; callers generating
    earlier forecast stages must pass as_of or supply a per-row as_of column.
    """
    history = _prepare_observations(observations, source_fixtures) if not observations.empty else None
    names = feature_names(metrics)
    output = []
    for target in targets.itertuples(index=False):
        kickoff = _timestamp(target.kickoff_utc, "target kickoff")
        default_cutoff = _timestamp(getattr(target, "kickoff_lower_bound_utc", kickoff), "kickoff lower bound") - pd.Timedelta(microseconds=1)
        cutoff = _timestamp(as_of if as_of is not None else getattr(target, "as_of", default_cutoff), "as_of")
        if cutoff >= _timestamp(getattr(target, "kickoff_lower_bound_utc", kickoff), "kickoff lower bound"):
            raise ValueError("Pre-match as_of must predate target kickoff")
        row = {"fixture_id": target.fixture_id, "as_of": cutoff.isoformat(), **{name: np.nan for name in names}}
        observed_times = []
        if history is not None:
            eligible = history.loc[(history.fixture_id != target.fixture_id) & (history.source_kickoff_utc < cutoff) & (history.observed_at <= cutoff)]
            if eligible.empty:
                row.update({name: 0.0 for name in names if "coverage" in name})
                row["feature_observed_at"] = None
                output.append(row)
                continue
            eligible = eligible.sort_values("observed_at", kind="stable").drop_duplicates(["fixture_id", "team_id"], keep="last")
            for side in ("home", "away"):
                team_id = getattr(target, f"{side}_team_id")
                for metric in metrics:
                    if metric.column not in eligible:
                        continue
                    for perspective in (("for", "against") if metric.perspective == "for_against" else (None,)):
                        if perspective == "against":
                            state = eligible.loc[(eligible.team_id != team_id) & ((eligible.home_team_id == team_id) | (eligible.away_team_id == team_id))]
                        else:
                            state = eligible.loc[eligible.team_id == team_id]
                        state = state.sort_values(["source_kickoff_utc", "fixture_id"], kind="stable")
                        values = pd.to_numeric(state[metric.column], errors="coerce")
                        stem = f"{side}_{metric.name}" + (f"_{perspective}" if perspective else "")
                        row[f"{stem}_coverage_l10"] = float(values.tail(10).notna().mean()) if len(values) else 0.0
                        if len(values):
                            row[f"{stem}_previous"] = values.iloc[-1]
                            row[f"{stem}_l5"] = values.tail(5).mean()
                            row[f"{stem}_l10"] = values.tail(10).mean()
                            for span in metric.spans:
                                row[f"{stem}_ewm{span}"] = values.ewm(span=span, adjust=False, min_periods=1).mean().iloc[-1]
                            observed_times.extend(state.loc[values.notna(), "observed_at"].tolist())
        row["feature_observed_at"] = max(observed_times).isoformat() if observed_times else None
        output.append(row)
    return pd.DataFrame(output, columns=["fixture_id", "as_of", *names, "feature_observed_at"])


def assert_point_in_time_features(frame: pd.DataFrame, *, observed_column: str = "feature_observed_at") -> None:
    if not {"fixture_id", "as_of", "kickoff_utc", observed_column}.issubset(frame):
        raise ValueError("Chronology audit requires fixture, cutoff, kickoff, and lineage")
    kickoff = pd.to_datetime(frame.kickoff_utc, utc=True, errors="raise")
    cutoff = pd.to_datetime(frame.as_of, utc=True, errors="raise")
    observed = pd.to_datetime(frame[observed_column], utc=True, errors="raise")
    if kickoff.isna().any() or cutoff.isna().any() or (cutoff >= kickoff).any() or (observed > cutoff).any():
        raise ValueError("Feature observations violate the forecast cutoff")
    feature_columns = [name for name in feature_names() if name in frame and "coverage" not in name]
    if feature_columns and (frame[feature_columns].notna().any(axis=1) & observed.isna()).any():
        raise ValueError("Populated provider features lack observation lineage")
