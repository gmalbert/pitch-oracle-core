"""Responsible, zero-default market assessment and bankroll simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketAssessment:
    model_probability: float
    market_probability: float
    offered_price: float
    fair_price: float
    edge: float
    expected_return: float


def assess_market(
    model_probability: float,
    market_probability: float,
    offered_price: float,
) -> MarketAssessment:
    values = (model_probability, market_probability, offered_price)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Market assessment values must be finite")
    if not 0 < model_probability < 1 or not 0 < market_probability < 1:
        raise ValueError("Probabilities must be strictly between zero and one")
    if offered_price <= 1:
        raise ValueError("Offered decimal price must exceed one")
    return MarketAssessment(
        model_probability=model_probability,
        market_probability=market_probability,
        offered_price=offered_price,
        fair_price=1.0 / model_probability,
        edge=model_probability - market_probability,
        expected_return=model_probability * offered_price - 1.0,
    )


@dataclass(frozen=True)
class StakePolicy:
    kelly_fraction: float = 0.25
    max_bet_fraction: float = 0.01
    max_fixture_fraction: float = 0.015
    max_team_fraction: float = 0.025
    max_league_fraction: float = 0.05
    minimum_edge: float = 0.03
    minimum_expected_return: float = 0.02

    def __post_init__(self) -> None:
        fractions = (
            self.kelly_fraction, self.max_bet_fraction, self.max_fixture_fraction,
            self.max_team_fraction, self.max_league_fraction,
        )
        if any(not 0 <= value <= 1 for value in fractions):
            raise ValueError("stake-policy fractions must be in [0, 1]")
        if self.max_bet_fraction > self.max_fixture_fraction:
            raise ValueError("single-bet cap cannot exceed fixture cap")


def kelly_fraction(probability: float, decimal_price: float) -> float:
    if not 0 <= probability <= 1 or decimal_price <= 1:
        raise ValueError("Invalid probability or decimal price")
    net = decimal_price - 1.0
    return max(0.0, (probability * decimal_price - 1.0) / net)


def recommended_stake_fraction(
    assessment: MarketAssessment,
    policy: StakePolicy,
    *,
    quote_is_fresh: bool,
    uncertainty_passed: bool,
    calibration_passed: bool,
    executable: bool = True,
) -> float:
    if not (
        quote_is_fresh and uncertainty_passed and calibration_passed and executable
    ):
        return 0.0
    if (
        assessment.edge < policy.minimum_edge
        or assessment.expected_return < policy.minimum_expected_return
    ):
        return 0.0
    return min(
        policy.max_bet_fraction,
        policy.kelly_fraction
        * kelly_fraction(assessment.model_probability, assessment.offered_price),
    )


def allocate_portfolio(
    opportunities: pd.DataFrame,
    policy: StakePolicy,
) -> pd.DataFrame:
    """Allocate candidates under fixture, team, and league exposure caps."""
    required = {
        "fixture_id", "team_id", "league_id", "model_probability",
        "market_probability", "decimal_price", "quote_is_fresh",
        "uncertainty_passed", "calibration_passed",
    }
    missing = required.difference(opportunities.columns)
    if missing:
        raise ValueError(f"Missing portfolio columns: {sorted(missing)}")
    frame = opportunities.copy()
    assessments = [
        assess_market(row.model_probability, row.market_probability, row.decimal_price)
        for row in frame.itertuples()
    ]
    frame["edge"] = [item.edge for item in assessments]
    frame["expected_return"] = [item.expected_return for item in assessments]
    frame = frame.sort_values(
        ["expected_return", "edge", "fixture_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    fixture_used: dict[str, float] = {}
    team_used: dict[str, float] = {}
    league_used: dict[str, float] = {}
    allocations: dict[object, tuple[float, str]] = {}
    for index, row in frame.iterrows():
        assessment = assess_market(
            float(row.model_probability), float(row.market_probability),
            float(row.decimal_price),
        )
        requested = recommended_stake_fraction(
            assessment,
            policy,
            quote_is_fresh=bool(row.quote_is_fresh),
            uncertainty_passed=bool(row.uncertainty_passed),
            calibration_passed=bool(row.calibration_passed),
            executable=bool(row.get("executable", True)),
        )
        fixture_id, team_id, league_id = map(
            str, (row.fixture_id, row.team_id, row.league_id)
        )
        available = min(
            requested,
            max(0.0, policy.max_fixture_fraction - fixture_used.get(fixture_id, 0.0)),
            max(0.0, policy.max_team_fraction - team_used.get(team_id, 0.0)),
            max(0.0, policy.max_league_fraction - league_used.get(league_id, 0.0)),
        )
        reason = "allocated" if available > 0 else (
            "safety_or_edge_gate" if requested == 0 else "exposure_cap"
        )
        allocations[index] = (available, reason)
        fixture_used[fixture_id] = fixture_used.get(fixture_id, 0.0) + available
        team_used[team_id] = team_used.get(team_id, 0.0) + available
        league_used[league_id] = league_used.get(league_id, 0.0) + available
    frame["stake_fraction"] = [allocations[index][0] for index in frame.index]
    frame["allocation_reason"] = [allocations[index][1] for index in frame.index]
    return frame.sort_index()


def bankroll_backtest(
    opportunities: pd.DataFrame,
    policy: StakePolicy,
    *,
    starting_bankroll: float = 1_000.0,
    transaction_cost_fraction: float = 0.0,
) -> pd.DataFrame:
    if starting_bankroll <= 0:
        raise ValueError("starting_bankroll must be positive")
    if not 0 <= transaction_cost_fraction < 1:
        raise ValueError("transaction_cost_fraction must be in [0, 1)")
    frame = opportunities.sort_values(["kickoff_utc", "fixture_id"]).copy()
    bankroll, peak = float(starting_bankroll), float(starting_bankroll)
    rows = []
    for opportunity in frame.itertuples():
        if hasattr(opportunity, "observed_at") and hasattr(opportunity, "issued_at"):
            if pd.Timestamp(opportunity.observed_at) >= pd.Timestamp(opportunity.issued_at):
                raise ValueError("Backtest quote was not available at forecast issue time")
        assessment = assess_market(
            opportunity.model_probability,
            opportunity.market_probability,
            opportunity.decimal_price,
        )
        fraction = recommended_stake_fraction(
            assessment,
            policy,
            quote_is_fresh=opportunity.quote_is_fresh,
            uncertainty_passed=opportunity.uncertainty_passed,
            calibration_passed=opportunity.calibration_passed,
            executable=getattr(opportunity, "executable", True),
        )
        stake = bankroll * fraction
        gross_profit = stake * (opportunity.decimal_price - 1) if opportunity.won else -stake
        transaction_cost = stake * transaction_cost_fraction
        profit = gross_profit - transaction_cost
        bankroll += profit
        peak = max(peak, bankroll)
        rows.append({
            "fixture_id": opportunity.fixture_id,
            "kickoff_utc": opportunity.kickoff_utc,
            "bankroll": bankroll,
            "stake": stake,
            "profit": profit,
            "gross_profit": gross_profit,
            "transaction_cost": transaction_cost,
            "drawdown": 1 - bankroll / peak,
            "edge": assessment.edge,
            "expected_return": assessment.expected_return,
            "model_probability": assessment.model_probability,
            "market_probability": assessment.market_probability,
            "decimal_price": assessment.offered_price,
            "won": bool(opportunity.won),
            "provider": getattr(opportunity, "provider", None),
            "season_id": getattr(opportunity, "season_id", None),
            "closing_line_value": getattr(opportunity, "closing_line_value", None),
        })
    return pd.DataFrame(rows)


def backtest_summary(
    ledger: pd.DataFrame,
    *,
    bootstrap_repetitions: int = 2_000,
    seed: int = 20260810,
) -> dict[str, object]:
    """Summarize return, risk, calibration, CLV, and bootstrap uncertainty."""
    if ledger.empty:
        raise ValueError("backtest ledger is empty")
    required = {"bankroll", "stake", "profit", "drawdown", "model_probability", "won"}
    missing = required.difference(ledger.columns)
    if missing:
        raise ValueError(f"Missing backtest summary columns: {sorted(missing)}")
    starting = float(ledger.iloc[0].bankroll - ledger.iloc[0].profit)
    turnover = float(ledger.stake.sum())
    profit = float(ledger.profit.sum())
    probability = ledger.model_probability.to_numpy(dtype=float)
    outcomes = ledger.won.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    roi_draws = np.zeros(bootstrap_repetitions, dtype=float)
    for index in range(bootstrap_repetitions):
        sample = rng.integers(0, len(ledger), len(ledger))
        sampled_stake = float(ledger.stake.to_numpy()[sample].sum())
        sampled_profit = float(ledger.profit.to_numpy()[sample].sum())
        roi_draws[index] = sampled_profit / sampled_stake if sampled_stake else 0.0
    clv = pd.to_numeric(
        ledger.get("closing_line_value", pd.Series(dtype=float)), errors="coerce"
    )
    by_season_provider = []
    grouping = [column for column in ("season_id", "provider") if column in ledger]
    if grouping:
        for keys, group in ledger.groupby(grouping, dropna=False, observed=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            group_turnover = float(group.stake.sum())
            by_season_provider.append({
                **dict(zip(grouping, keys)),
                "bets": len(group),
                "turnover": group_turnover,
                "roi": float(group.profit.sum() / group_turnover) if group_turnover else 0.0,
            })
    return {
        "starting_bankroll": starting,
        "ending_bankroll": float(ledger.iloc[-1].bankroll),
        "turnover": turnover,
        "profit": profit,
        "transaction_costs": float(ledger.get("transaction_cost", pd.Series(0.0, index=ledger.index)).sum()),
        "roi": profit / turnover if turnover else 0.0,
        "maximum_drawdown": float(ledger.drawdown.max()),
        "calibration_brier": float(np.mean(np.square(probability - outcomes))),
        "mean_closing_line_value": float(clv.mean()) if clv.notna().any() else None,
        "roi_lower_95": float(np.quantile(roi_draws, 0.025)),
        "roi_upper_95": float(np.quantile(roi_draws, 0.975)),
        "by_season_provider": by_season_provider,
    }
