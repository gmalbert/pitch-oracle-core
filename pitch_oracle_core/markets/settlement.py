"""Exact Asian total and handicap settlement including quarter lines."""

from __future__ import annotations

from decimal import Decimal

from pitch_oracle_core.domain.probability_grid import ProbabilityGrid


def split_quarter_line(line: Decimal) -> tuple[Decimal, Decimal]:
    quarter = line * 4
    if quarter != quarter.to_integral_value():
        raise ValueError("line must be on a quarter-goal increment")
    # Odd quarter counts are split lines for both positive and negative
    # handicaps/totals (for example -0.25 -> -0.50 and 0.00).
    if abs(int(quarter)) % 2 == 1:
        return line - Decimal("0.25"), line + Decimal("0.25")
    return line, line


def over_leg_net_return(
    total_goals: int, line: Decimal, decimal_odds: Decimal
) -> Decimal:
    if total_goals < 0 or decimal_odds <= 1:
        raise ValueError("invalid goals or decimal odds")
    total = Decimal(total_goals)
    if total > line:
        return decimal_odds - 1
    if total == line:
        return Decimal(0)
    return Decimal(-1)


def over_quarter_net_return(
    total_goals: int, line: Decimal, decimal_odds: Decimal
) -> Decimal:
    left, right = split_quarter_line(line)
    return (
        over_leg_net_return(total_goals, left, decimal_odds)
        + over_leg_net_return(total_goals, right, decimal_odds)
    ) / 2


def under_quarter_net_return(
    total_goals: int, line: Decimal, decimal_odds: Decimal
) -> Decimal:
    left, right = split_quarter_line(line)

    def leg(component: Decimal) -> Decimal:
        total = Decimal(total_goals)
        if total < component:
            return decimal_odds - 1
        if total == component:
            return Decimal(0)
        return Decimal(-1)

    return (leg(left) + leg(right)) / 2


def handicap_quarter_net_return(
    home_goals: int,
    away_goals: int,
    home_handicap: Decimal,
    decimal_odds: Decimal,
) -> Decimal:
    if decimal_odds <= 1:
        raise ValueError("decimal odds must exceed one")
    left, right = split_quarter_line(home_handicap)

    def leg(component: Decimal) -> Decimal:
        adjusted = Decimal(home_goals - away_goals) + component
        if adjusted > 0:
            return decimal_odds - 1
        if adjusted == 0:
            return Decimal(0)
        return Decimal(-1)

    return (leg(left) + leg(right)) / 2


def expected_over_net_return(
    grid: ProbabilityGrid,
    line: Decimal,
    decimal_odds: Decimal,
    *,
    max_tail: float = 1e-8,
) -> float:
    if grid.tail_mass > max_tail:
        raise ValueError("grid tail is too large for point expected return")
    expected = Decimal(0)
    for home_goals in range(grid.mass.shape[0]):
        for away_goals in range(grid.mass.shape[1]):
            probability = Decimal(str(grid.mass[home_goals, away_goals]))
            expected += probability * over_quarter_net_return(
                home_goals + away_goals, line, decimal_odds
            )
    return float(expected)
