# Optional context and market implementation

Optional data must obey the same point-in-time and provenance rules as baseline data.
This chapter supplies the concrete contracts and calculations for F06, F11, F22–F25,
F49, and F50. None of these inputs are required to produce the no-odds forecast.

## 1. Timestamped contextual snapshots

Never merge “current” squad, manager, referee, or weather data onto historical
fixtures. Every observation has an effective interval and an observation time.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar


Payload = TypeVar("Payload")


@dataclass(frozen=True)
class Snapshot(Generic[Payload]):
    entity_id: str
    observed_at: datetime
    effective_from: datetime
    effective_to: datetime | None
    source: str
    source_event_id: str
    payload: Payload

    def usable_for(self, kickoff_utc: datetime) -> bool:
        return (
            self.observed_at < kickoff_utc
            and self.effective_from <= kickoff_utc
            and (self.effective_to is None or kickoff_utc < self.effective_to)
        )


def latest_usable_snapshot(snapshots, kickoff_utc):
    usable = [item for item in snapshots if item.usable_for(kickoff_utc)]
    return max(usable, key=lambda item: item.observed_at) if usable else None
```

The source observation—not the effective event date—is the leakage boundary. An
injury retrospectively reported after kickoff cannot become a historical pre-match
feature for that fixture.

## 2. Schedule, travel, and recovery load (F22)

Use all-competition fixtures when available. Domestic-league-only rest remains a
clearly labeled fallback.

```python
from math import asin, cos, radians, sin, sqrt
import numpy as np
import pandas as pd


def haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_km = 6371.0088
    d_lat = radians(lat_b - lat_a)
    d_lon = radians(lon_b - lon_a)
    a = (
        sin(d_lat / 2) ** 2
        + cos(radians(lat_a)) * cos(radians(lat_b)) * sin(d_lon / 2) ** 2
    )
    return 2 * radius_km * asin(sqrt(a))


def recovery_features(team_fixtures: pd.DataFrame) -> pd.DataFrame:
    frame = team_fixtures.sort_values(["team_id", "kickoff_utc"]).copy()
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True)
    group = frame.groupby("team_id", sort=False)
    frame["days_since_previous"] = group.kickoff_utc.diff().dt.total_seconds() / 86_400

    def count_previous_14d(series: pd.Series) -> pd.Series:
        values = series.astype("int64").to_numpy()
        horizon_ns = 14 * 86_400 * 1_000_000_000
        counts = [
            index - np.searchsorted(values, value - horizon_ns, side="left")
            for index, value in enumerate(values)
        ]
        return pd.Series(counts, index=series.index, dtype=int)

    frame["matches_previous_14d"] = group.kickoff_utc.transform(count_previous_14d)
    frame["short_rest"] = frame.days_since_previous < 4
    frame["travel_km"] = frame.apply(
        lambda row: haversine_km(
            row.previous_venue_lat, row.previous_venue_lon,
            row.venue_lat, row.venue_lon,
        ) if pd.notna(row.previous_venue_lat) and pd.notna(row.venue_lat) else float("nan"),
        axis=1,
    )
    frame["recovery_load"] = (
        frame["short_rest"].astype(float)
        + frame["matches_previous_14d"].clip(lower=0).div(5)
        + frame["travel_km"].fillna(0).div(2_000).clip(upper=1)
    )
    return frame
```

In production, use a time-indexed `Series.rolling("14D")` per team rather than
counting final-season fixtures. Publish venue-coordinate confidence and omit travel
when the prior/current venue is unresolved.

## 3. Squad availability impact (F24)

The baseline is minutes-weighted player contribution with replacement shrinkage. A
provider with richer on/off or plus-minus estimates can populate the same contract.

```python
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class PlayerAvailability:
    player_id: str
    team_id: str
    status: str                  # available, doubtful, out, suspended
    availability_probability: float
    observed_at: str
    source: str


def squad_absence_impact(
    availability: pd.DataFrame,
    player_strength: pd.DataFrame,
    *,
    replacement_level: float = -0.25,
) -> pd.DataFrame:
    required = {"player_id", "team_id", "availability_probability"}
    if not required.issubset(availability.columns):
        raise ValueError("Availability artifact is incomplete")
    joined = availability.merge(
        player_strength[["player_id", "minutes_share", "strength_per90"]],
        on="player_id", how="left", validate="many_to_one",
    )
    # Missing player quality is explicitly shrunk to replacement level.
    joined["strength_per90"] = joined.strength_per90.fillna(replacement_level)
    joined["minutes_share"] = joined.minutes_share.fillna(0).clip(0, 1)
    joined["expected_absence"] = 1 - joined.availability_probability.clip(0, 1)
    joined["impact"] = (
        joined.expected_absence
        * joined.minutes_share
        * (joined.strength_per90 - replacement_level)
    )
    return joined.groupby("team_id", as_index=False).agg(
        expected_missing_strength=("impact", "sum"),
        players_flagged=("expected_absence", lambda series: int((series > 0.25).sum())),
    )
```

Show the impact as a range driven by availability probability. Do not interpret an
unavailable provider as a healthy squad.

## 4. Manager tenure and change effects (F23)

Manager identity is valid over an interval. Estimate a “bounce” with partial pooling,
not a raw before/after win-rate comparison.

```python
import numpy as np
import pandas as pd


def attach_manager_at_kickoff(matches: pd.DataFrame, tenures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ("home", "away"):
        left = matches[["fixture_id", "kickoff_utc", f"{side}_team_id"]].rename(
            columns={f"{side}_team_id": "team_id"}
        ).sort_values("kickoff_utc")
        right = tenures.sort_values("effective_from")
        merged = pd.merge_asof(
            left, right,
            left_on="kickoff_utc", right_on="effective_from",
            by="team_id", direction="backward",
        )
        valid = merged.effective_to.isna() | (merged.kickoff_utc < merged.effective_to)
        merged.loc[~valid, "manager_id"] = pd.NA
        rows.append(merged[["fixture_id", "manager_id"]].rename(
            columns={"manager_id": f"{side}_manager_id"}
        ))
    return rows[0].merge(rows[1], on="fixture_id", validate="one_to_one")


def shrunk_manager_effect(
    residual_points: pd.Series,
    *,
    prior_mean: float = 0.0,
    prior_matches: float = 12.0,
) -> float:
    values = residual_points.dropna().to_numpy(dtype=float)
    return float(
        (values.sum() + prior_mean * prior_matches) / (len(values) + prior_matches)
    )
```

`residual_points` should be actual points minus pre-match expected points, which
controls for schedule strength. Report the effect interval and tenure sample.

## 5. Referee matchup with empirical shrinkage (F25)

```python
def beta_binomial_rate(
    successes: float, trials: float, league_rate: float, prior_matches: float = 20.0
) -> float:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Invalid binomial counts")
    alpha = league_rate * prior_matches
    beta = (1 - league_rate) * prior_matches
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
    output["penalty_rate_shrunk"] = output.apply(
        lambda row: beta_binomial_rate(
            row.penalties, row.matches,
            league.penalties.sum() / max(league.matches.sum(), 1),
        ), axis=1,
    )
    return output
```

Cards are counts rather than binary events, so the cards formula uses a Gamma/Poisson-
style exposure prior; penalties use a Beta/Binomial prior. All aggregates must be
calculated only from assignments before the forecast fixture.

## 6. Forecast revision timeline (F11)

Forecast rows are immutable. A new issue time creates a new revision.

```python
REVISION_LABELS = (
    (168, "initial"),
    (24, "day_before"),
    (2, "pre_lineup"),
    (0, "lineup"),
)


def revision_label(kickoff_utc, issued_at) -> str:
    hours = (kickoff_utc - issued_at).total_seconds() / 3600
    for threshold, label in REVISION_LABELS:
        if hours >= threshold:
            return label
    return "late_or_live"


def forecast_revision_deltas(ledger: pd.DataFrame) -> pd.DataFrame:
    frame = ledger.sort_values(["fixture_id", "issued_at"]).copy()
    for column in ("p_home", "p_draw", "p_away", "expected_home_goals", "expected_away_goals"):
        frame[f"delta_{column}"] = frame.groupby("fixture_id")[column].diff()
    return frame
```

The application should refuse to label anything issued at or after kickoff as a
pre-match forecast.

## 7. Timestamped odds snapshots (F49)

```python
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class OddsQuote:
    fixture_id: str
    market: str                 # "1x2", "total_2_5", "btts"
    selection: str
    decimal_price: float
    bookmaker: str
    observed_at: datetime
    source: str
    executable: bool = False

    def __post_init__(self) -> None:
        if self.decimal_price <= 1:
            raise ValueError("Decimal price must exceed 1")
```

Store every observation; do not overwrite the opener with the latest price. The
market page selects the last quote before its requested snapshot time.

## 8. De-vigging and consensus fair probabilities (F49)

The multiplicative method is transparent and stable. Add power/Shin methods later as
explicit choices, not silent replacements.

```python
import numpy as np
import pandas as pd


def multiplicative_devig(decimal_prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(decimal_prices, dtype=float)
    if prices.ndim != 1 or len(prices) < 2 or (prices <= 1).any():
        raise ValueError("A market needs at least two valid decimal prices")
    implied = 1.0 / prices
    return implied / implied.sum()


def bookmaker_market_probabilities(quotes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["fixture_id", "market", "bookmaker", "observed_at"]
    for key, group in quotes.groupby(keys, observed=True):
        group = group.sort_values("selection")
        fair = multiplicative_devig(group.decimal_price.to_numpy())
        for row, probability in zip(group.itertuples(), fair):
            rows.append({
                **dict(zip(keys, key)),
                "selection": row.selection,
                "decimal_price": row.decimal_price,
                "fair_probability": float(probability),
            })
    return pd.DataFrame(rows)


def consensus_market(bookmaker_probabilities: pd.DataFrame) -> pd.DataFrame:
    # Median is robust to a stale or malformed outlying book.
    return bookmaker_probabilities.groupby(
        ["fixture_id", "market", "selection"], as_index=False
    ).agg(
        consensus_probability=("fair_probability", "median"),
        market_low=("fair_probability", "min"),
        market_high=("fair_probability", "max"),
        books=("bookmaker", "nunique"),
        best_price=("decimal_price", "max"),
        observed_at=("observed_at", "max"),
    )
```

Display dispersion and book count. A one-book “consensus” is labeled single-source.

## 9. Model fair price, edge, and expected value (F49)

```python
from dataclasses import dataclass
import math


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
```

Require quote freshness, minimum books, entity match, calibration cohort health, and
uncertainty bounds before calling an assessment actionable.

## 10. Constrained fractional Kelly (F50)

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class StakePolicy:
    kelly_fraction: float = 0.25
    max_bet_fraction: float = 0.01
    max_fixture_fraction: float = 0.015
    minimum_edge: float = 0.03
    minimum_expected_return: float = 0.02


def kelly_fraction(probability: float, decimal_price: float) -> float:
    if not 0 <= probability <= 1 or decimal_price <= 1:
        raise ValueError("Invalid probability or decimal price")
    net = decimal_price - 1.0
    raw = (probability * decimal_price - 1.0) / net
    return max(0.0, raw)


def recommended_stake_fraction(
    assessment: MarketAssessment,
    policy: StakePolicy,
    *,
    quote_is_fresh: bool,
    uncertainty_passed: bool,
    calibration_passed: bool,
) -> float:
    if not (quote_is_fresh and uncertainty_passed and calibration_passed):
        return 0.0
    if (
        assessment.edge < policy.minimum_edge
        or assessment.expected_return < policy.minimum_expected_return
    ):
        return 0.0
    return min(
        policy.max_bet_fraction,
        policy.kelly_fraction * kelly_fraction(
            assessment.model_probability, assessment.offered_price
        ),
    )
```

Before portfolio publication, cap total fixture, team, league, kickoff-window, and
correlated-market exposure. Over 2.5 and BTTS in the same fixture are not independent.

## 11. Walk-forward bankroll backtest (F50)

```python
def bankroll_backtest(
    opportunities: pd.DataFrame,
    policy: StakePolicy,
    *,
    starting_bankroll: float = 1_000.0,
) -> pd.DataFrame:
    frame = opportunities.sort_values(["kickoff_utc", "fixture_id"]).copy()
    bankroll = float(starting_bankroll)
    peak = bankroll
    rows = []
    for opportunity in frame.itertuples():
        assessment = assess_market(
            opportunity.model_probability,
            opportunity.market_probability,
            opportunity.decimal_price,
        )
        fraction = recommended_stake_fraction(
            assessment, policy,
            quote_is_fresh=opportunity.quote_is_fresh,
            uncertainty_passed=opportunity.uncertainty_passed,
            calibration_passed=opportunity.calibration_passed,
        )
        stake = bankroll * fraction
        profit = stake * (opportunity.decimal_price - 1) if opportunity.won else -stake
        bankroll += profit
        peak = max(peak, bankroll)
        rows.append({
            "fixture_id": opportunity.fixture_id,
            "kickoff_utc": opportunity.kickoff_utc,
            "bankroll": bankroll,
            "stake": stake,
            "profit": profit,
            "drawdown": 1 - bankroll / peak,
            "edge": assessment.edge,
            "expected_return": assessment.expected_return,
        })
    return pd.DataFrame(rows)
```

Backtest with the quote available at the forecast issue time, not the closing price
unless the strategy explicitly issues at close. Report turnover, ROI, maximum
drawdown, calibration, closing-line value, bootstrap uncertainty, and results by
season/provider—not just final profit.

## 12. Context capability matrix

| Capability state | Forecast behavior | UI behavior |
|---|---|---|
| Available and fresh | Feature may enter an explicitly trained context model. | Render data, source, timestamp, and forecast delta. |
| Degraded coverage | Use only rows meeting the model's coverage contract; otherwise base forecast. | Render coverage warning and affected entities. |
| Stale | Exclude from fresh inference. | Show last observation and stale label. |
| Unavailable | Base no-odds forecast remains valid. | Hide optional page or show one honest unavailable card. |

## 13. Context and market tests

Add tests for:

- observations at/after kickoff are never selected;
- manager/referee/availability validity intervals;
- accent/alias resolution before provider joins;
- haversine symmetry and zero-distance behavior;
- future scheduled fixtures do not contaminate outcome form;
- de-vigged probabilities sum to one per bookmaker market;
- consensus remains stable with one extreme outlier;
- price freshness and executable flags;
- no negative Kelly stake and all safety gates default to zero;
- fixture/team/league correlation exposure caps;
- walk-forward bankroll uses only prices available at issue time;
- optional provider failure leaves the base forecast artifact valid.
