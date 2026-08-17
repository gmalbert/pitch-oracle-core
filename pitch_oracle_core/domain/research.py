"""Disposition registry for all 26 evidence-backed research initiatives."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchInitiative:
    initiative_id: str
    name: str
    disposition: str
    implementation: str


_RESEARCH = (
    ("Common ScoreModel/probability-grid API", "adopt", "domain.probability_grid + models.protocol"),
    ("Count-distribution residual report", "adopt", "evaluation.distribution_diagnostics"),
    ("Tuned recency/half-life", "adopt", "evaluation.recency"),
    ("Bivariate Poisson challenger", "experiment", "models.distribution_registry"),
    ("Diagonal-inflated/hurdle challenger", "experiment", "models.distribution_registry"),
    ("NB/CMP dispersion challenger", "experiment", "models.distribution_registry"),
    ("Weibull/copula challenger", "deferred", "registered_deferred_candidate"),
    ("Score-driven/state-space strengths", "experiment", "models.dynamic_strength"),
    ("Hierarchical multi-league priors", "experiment", "models.hierarchical_prior"),
    ("Elo/Pi/Glicko rating tournament", "experiment", "models.ratings"),
    ("Rank-plus-covariate goals model", "experiment", "models.rank_covariate"),
    ("Player-strength lineup prior", "capability_gated", "players.lineup_strength"),
    ("Independent vs market-aware tracks", "adopt", "models.protocol"),
    ("Multi-method de-vig engine", "adopt", "markets.devig"),
    ("Market-implied expected-goals comparator", "experiment", "markets.benchmark"),
    ("Proper-score panel", "adopt", "evaluation.scores"),
    ("CORP reliability and sharpness view", "adopt", "evaluation.calibration"),
    ("Paired block-bootstrap model deltas", "adopt", "evaluation.paired_tests"),
    ("White/SPA multi-candidate control", "adopt", "evaluation.paired_tests"),
    ("Immutable forward-test ledger", "adopt", "evaluation.rolling_origin"),
    ("Exact quarter-line settlement", "adopt", "markets.settlement"),
    ("Closing-price and CLV audit", "capability_gated", "markets.benchmark"),
    ("Provider-neutral shot xG", "capability_gated", "events.schema + providers"),
    ("xT/VAEP team/player snapshots", "capability_gated", "events.value"),
    ("Tracking/pitch-control lab", "research_only", "events.pitch_control"),
    ("Uncertainty-aware capped portfolio simulator", "research_only", "markets.portfolio"),
)


def research_initiatives() -> tuple[ResearchInitiative, ...]:
    return tuple(
        ResearchInitiative(f"R{index:02d}", name, disposition, implementation)
        for index, (name, disposition, implementation) in enumerate(_RESEARCH, start=1)
    )


def validate_research_registry(items: tuple[ResearchInitiative, ...]) -> None:
    expected = {f"R{index:02d}" for index in range(1, 27)}
    if {item.initiative_id for item in items} != expected or len(items) != 26:
        raise ValueError("research registry must contain exactly R01 through R26")
    if any(not item.implementation.strip() for item in items):
        raise ValueError("every research initiative requires an implementation route")
