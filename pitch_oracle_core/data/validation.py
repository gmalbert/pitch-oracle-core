"""Publication-blocking and informational data-quality gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityCheck:
    check: str
    status: str
    severity: str
    observed: object
    expected: object
    message: str = ""


@dataclass(frozen=True)
class QualityReport:
    checks: tuple[QualityCheck, ...]

    @property
    def publishable(self) -> bool:
        return not any(
            check.severity == "blocking" and check.status != "passed"
            for check in self.checks
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "publishable": self.publishable,
            "checks": [asdict(check) for check in self.checks],
        }

    def require_publishable(self) -> None:
        if not self.publishable:
            failed = [
                check.check
                for check in self.checks
                if check.severity == "blocking" and check.status != "passed"
            ]
            raise RuntimeError("Publication blocked by: " + ", ".join(failed))

    @classmethod
    def combine(cls, *reports: "QualityReport") -> "QualityReport":
        return cls(tuple(check for report in reports for check in report.checks))


def require_quality_report_payload(report: dict[str, object]) -> None:
    """Prevent manifest publication from bypassing a serialized blocking check."""
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        raise RuntimeError("Publication blocked: quality report has no checks")
    failed = [
        str(check.get("check", "unknown"))
        for check in checks
        if isinstance(check, dict)
        and check.get("severity") == "blocking"
        and check.get("status") != "passed"
    ]
    declared = report.get("publishable") is True
    if failed or not declared:
        reason = ", ".join(failed) if failed else "publishable flag is not true"
        raise RuntimeError("Publication blocked by: " + reason)


def validate_canonical_fixtures(fixtures: pd.DataFrame) -> QualityReport:
    required = {
        "fixture_id", "edition_id", "rules_version", "kickoff_utc",
        "home_team_id", "away_team_id",
        "provider_event_id", "source", "observed_at",
    }
    checks: list[QualityCheck] = []
    missing = sorted(required.difference(fixtures.columns))
    checks.append(QualityCheck(
        "fixture_schema", "passed" if not missing else "failed", "blocking",
        missing, [], "Missing required canonical fixture fields" if missing else "",
    ))
    if missing:
        return QualityReport(tuple(checks))
    duplicate_ids = int(fixtures["fixture_id"].duplicated().sum())
    duplicate_provider = int(fixtures.duplicated(["source", "provider_event_id"]).sum())
    unresolved = int(
        fixtures[["home_team_id", "away_team_id"]].isna().any(axis=1).sum()
    )
    kickoff = pd.to_datetime(fixtures["kickoff_utc"], utc=True, errors="coerce")
    observed = pd.to_datetime(fixtures["observed_at"], utc=True, errors="coerce")
    invalid_times = int((kickoff.isna() | observed.isna()).sum())
    same_team = int((fixtures["home_team_id"] == fixtures["away_team_id"]).sum())
    missing_edition = int(
        fixtures[["edition_id", "rules_version"]]
        .replace("", np.nan)
        .isna()
        .any(axis=1)
        .sum()
    )
    for name, observed_value in (
        ("duplicate_fixture_ids", duplicate_ids),
        ("duplicate_provider_events", duplicate_provider),
        ("unresolved_active_teams", unresolved),
        ("invalid_timestamps", invalid_times),
        ("self_fixtures", same_team),
        ("missing_edition_or_rules", missing_edition),
    ):
        checks.append(QualityCheck(
            name, "passed" if observed_value == 0 else "failed", "blocking",
            observed_value, 0,
        ))
    return QualityReport(tuple(checks))


def validate_forecast_artifacts(
    fixtures: pd.DataFrame,
    forecasts: pd.DataFrame,
    scorelines: pd.DataFrame,
) -> None:
    scheduled = fixtures.loc[fixtures["status"] == "scheduled", "fixture_id"]
    if set(scheduled) != set(forecasts["fixture_id"]):
        raise ValueError("Scheduled fixture and forecast IDs disagree")
    probabilities = forecasts[["p_home", "p_draw", "p_away"]]
    if not np.isfinite(probabilities.to_numpy()).all():
        raise ValueError("Forecast probabilities must be finite")
    if (probabilities.to_numpy() < 0).any():
        raise ValueError("Forecast probabilities cannot be negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("1X2 probabilities do not sum to one")
    mass = scorelines.groupby("fixture_id")["probability"].sum()
    if not np.allclose(mass.reindex(scheduled), 1.0, atol=1e-6):
        raise ValueError("Scoreline mass does not sum to one")
    derived_rows = []
    for fixture_id, group in scorelines.groupby("fixture_id", observed=True):
        derived_rows.append({
            "fixture_id": fixture_id,
            "p_home": group.loc[group.home_goals > group.away_goals, "probability"].sum(),
            "p_draw": group.loc[group.home_goals == group.away_goals, "probability"].sum(),
            "p_away": group.loc[group.home_goals < group.away_goals, "probability"].sum(),
        })
    derived = pd.DataFrame(derived_rows).set_index("fixture_id")
    joined = forecasts.set_index("fixture_id").join(
        derived, lsuffix="_forecast", rsuffix="_scores"
    )
    for outcome in ("home", "draw", "away"):
        if not np.allclose(
            joined[f"p_{outcome}_forecast"], joined[f"p_{outcome}_scores"], atol=1e-6
        ):
            raise ValueError(f"Scoreline and 1X2 {outcome} probabilities disagree")


def validate_pre_match_features(
    features: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    timestamp_columns: tuple[str, ...] = ("feature_timestamp", "observed_at"),
    numeric_columns: tuple[str, ...] | None = None,
) -> QualityReport:
    """Prove that persisted model inputs existed strictly before kickoff."""
    checks: list[QualityCheck] = []
    if "fixture_id" not in features or "fixture_id" not in fixtures:
        return QualityReport((QualityCheck(
            "feature_fixture_key", "failed", "blocking", "missing", "fixture_id",
        ),))
    present_timestamps = [column for column in timestamp_columns if column in features]
    if not present_timestamps:
        return QualityReport((QualityCheck(
            "feature_timestamps", "failed", "blocking", [], list(timestamp_columns),
            "No point-in-time timestamp is present on the feature artifact",
        ),))
    kickoff = fixtures[["fixture_id", "kickoff_utc"]].drop_duplicates("fixture_id")
    joined = features.merge(kickoff, on="fixture_id", how="left", validate="many_to_one")
    kickoff_values = pd.to_datetime(joined["kickoff_utc"], utc=True, errors="coerce")
    missing_fixture = int(kickoff_values.isna().sum())
    checks.append(QualityCheck(
        "feature_fixture_coverage", "passed" if missing_fixture == 0 else "failed",
        "blocking", missing_fixture, 0,
    ))
    for column in present_timestamps:
        observed = pd.to_datetime(joined[column], utc=True, errors="coerce")
        invalid = int((observed.isna() | (observed >= kickoff_values)).sum())
        checks.append(QualityCheck(
            f"{column}_before_kickoff", "passed" if invalid == 0 else "failed",
            "blocking", invalid, 0,
        ))
    selected_numeric = list(numeric_columns or tuple(
        column for column in features.select_dtypes(include=[np.number]).columns
        if column not in {"home_goals", "away_goals"}
    ))
    non_finite = 0
    if selected_numeric:
        values = features[selected_numeric].to_numpy(dtype=float)
        non_finite = int((~np.isfinite(values)).sum())
    checks.append(QualityCheck(
        "finite_model_inputs", "passed" if non_finite == 0 else "failed",
        "blocking", non_finite, 0,
    ))
    return QualityReport(tuple(checks))


def validate_match_values(
    fixtures: pd.DataFrame, odds: pd.DataFrame | None = None
) -> QualityReport:
    checks: list[QualityCheck] = []
    goal_columns = [column for column in ("home_goals", "away_goals") if column in fixtures]
    negative_goals = int(
        sum((pd.to_numeric(fixtures[column], errors="coerce").dropna() < 0).sum()
            for column in goal_columns)
    )
    checks.append(QualityCheck(
        "non_negative_goals", "passed" if negative_goals == 0 else "failed",
        "blocking", negative_goals, 0,
    ))
    if odds is not None:
        price_column = next(
            (column for column in ("decimal_price", "decimal_odds", "price") if column in odds),
            None,
        )
        if price_column is None:
            invalid_prices = len(odds)
        else:
            prices = pd.to_numeric(odds[price_column], errors="coerce")
            invalid_prices = int((prices.isna() | ~np.isfinite(prices) | (prices <= 1)).sum())
        checks.append(QualityCheck(
            "valid_decimal_odds", "passed" if invalid_prices == 0 else "failed",
            "blocking", invalid_prices, 0,
        ))
    return QualityReport(tuple(checks))


def validate_forecast_metadata(
    fixtures: pd.DataFrame, forecasts: pd.DataFrame
) -> QualityReport:
    required = {
        "fixture_id", "issued_at", "effective_sample_size", "cold_start_status"
    }
    missing = sorted(required.difference(forecasts.columns))
    if missing:
        return QualityReport((QualityCheck(
            "forecast_metadata", "failed", "blocking", missing, [],
            "Forecasts require issue time, effective sample size, and cold-start status",
        ),))
    joined = forecasts.merge(
        fixtures[["fixture_id", "kickoff_utc"]], on="fixture_id", how="left",
        validate="many_to_one",
    )
    issued = pd.to_datetime(joined["issued_at"], utc=True, errors="coerce")
    kickoff = pd.to_datetime(joined["kickoff_utc"], utc=True, errors="coerce")
    invalid_time = int((issued.isna() | kickoff.isna() | (issued >= kickoff)).sum())
    sample = pd.to_numeric(joined["effective_sample_size"], errors="coerce")
    invalid_sample = int((sample.isna() | ~np.isfinite(sample) | (sample < 0)).sum())
    missing_status = int(
        joined["cold_start_status"].replace("", np.nan).isna().sum()
    )
    return QualityReport((
        QualityCheck("forecast_issued_before_kickoff", "passed" if invalid_time == 0 else "failed", "blocking", invalid_time, 0),
        QualityCheck("forecast_effective_sample", "passed" if invalid_sample == 0 else "failed", "blocking", invalid_sample, 0),
        QualityCheck("forecast_cold_start_status", "passed" if missing_status == 0 else "failed", "blocking", missing_status, 0),
    ))


def validate_publication_bundle(
    *,
    fixtures: pd.DataFrame,
    forecasts: pd.DataFrame,
    scorelines: pd.DataFrame,
    pre_match_features: pd.DataFrame,
    odds: pd.DataFrame | None = None,
) -> QualityReport:
    """Run the P0 publication blockers as one composable report."""
    reports = [
        validate_canonical_fixtures(fixtures),
        validate_match_values(fixtures, odds),
        validate_pre_match_features(pre_match_features, fixtures),
        validate_forecast_metadata(fixtures, forecasts),
    ]
    try:
        validate_forecast_artifacts(fixtures, forecasts, scorelines)
    except (KeyError, TypeError, ValueError) as exc:
        coherence = QualityReport((QualityCheck(
            "forecast_coherence", "failed", "blocking", str(exc), "coherent",
        ),))
    else:
        coherence = QualityReport((QualityCheck(
            "forecast_coherence", "passed", "blocking", "coherent", "coherent",
        ),))
    return QualityReport.combine(*reports, coherence)
