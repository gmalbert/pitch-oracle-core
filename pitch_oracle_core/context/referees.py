"""Empirically shrunk referee discipline profiles."""

from __future__ import annotations

import pandas as pd


def attach_referee_at_kickoff(
    fixtures: pd.DataFrame, assignments: pd.DataFrame
) -> pd.DataFrame:
    """Select only referee assignments observed and valid before kickoff."""
    required = {"fixture_id", "referee_id", "observed_at"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Missing referee-assignment columns: {sorted(missing)}")
    left = fixtures[["fixture_id", "kickoff_utc"]].copy()
    left["kickoff_utc"] = pd.to_datetime(left["kickoff_utc"], utc=True)
    right = assignments.copy()
    right["observed_at"] = pd.to_datetime(right["observed_at"], utc=True)
    right["effective_from"] = pd.to_datetime(
        right.get("effective_from", right["observed_at"]), utc=True
    )
    right["effective_to"] = pd.to_datetime(
        right.get("effective_to", pd.Series(pd.NaT, index=right.index)), utc=True
    )
    joined = left.merge(right, on="fixture_id", how="left")
    valid = (
        (joined.observed_at < joined.kickoff_utc)
        & (joined.effective_from <= joined.kickoff_utc)
        & (joined.effective_to.isna() | (joined.kickoff_utc < joined.effective_to))
    )
    selected = (
        joined.loc[valid]
        .sort_values(["fixture_id", "observed_at"])
        .drop_duplicates("fixture_id", keep="last")
    )
    return left.merge(
        selected[["fixture_id", "referee_id", "observed_at"]],
        on="fixture_id", how="left", validate="one_to_one",
    )


def beta_binomial_rate(
    successes: float,
    trials: float,
    league_rate: float,
    prior_matches: float = 20.0,
) -> float:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Invalid binomial counts")
    if not 0 <= league_rate <= 1 or prior_matches <= 0:
        raise ValueError("Invalid prior")
    alpha, beta = league_rate * prior_matches, (1 - league_rate) * prior_matches
    return float((successes + alpha) / (trials + alpha + beta))


def referee_profile(assignments: pd.DataFrame, league: pd.DataFrame) -> pd.DataFrame:
    league_card_rate = league.cards.sum() / max(league.matches.sum(), 1)
    output = assignments.groupby("referee_id", as_index=False).agg(
        matches=("fixture_id", "nunique"),
        cards=("cards", "sum"),
        penalties=("penalties", "sum"),
        fouls=("fouls", "sum"),
    )
    output["cards_per_match_shrunk"] = (
        output.cards + league_card_rate * 20
    ) / (output.matches + 20)
    league_penalty_rate = league.penalties.sum() / max(league.matches.sum(), 1)
    output["penalty_rate_shrunk"] = output.apply(
        lambda row: beta_binomial_rate(
            row.penalties, row.matches, league_penalty_rate
        ),
        axis=1,
    )
    return output
