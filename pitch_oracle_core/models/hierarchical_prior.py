"""Transparent empirical partial-pooling priors for cold-start clubs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrengthPrior:
    attack_mean: float
    defense_mean: float
    effective_matches: float
    provenance: str


def partial_pool_strength(
    observed_mean: float,
    observed_matches: float,
    prior_mean: float,
    prior_matches: float,
) -> tuple[float, float]:
    if observed_matches < 0 or prior_matches <= 0:
        raise ValueError("invalid effective sample sizes")
    weight = observed_matches / (observed_matches + prior_matches)
    return weight * observed_mean + (1 - weight) * prior_mean, weight


def promoted_team_prior(
    lower_division_attack: float | None,
    lower_division_defense: float | None,
    league_attack_mean: float,
    league_defense_mean: float,
    translation_weight: float,
) -> StrengthPrior:
    if not 0 <= translation_weight <= 1:
        raise ValueError("translation_weight must be in [0, 1]")
    if lower_division_attack is None or lower_division_defense is None:
        return StrengthPrior(
            league_attack_mean,
            league_defense_mean,
            effective_matches=0.0,
            provenance="league_prior_no_lower_division_evidence",
        )
    return StrengthPrior(
        translation_weight * lower_division_attack
        + (1 - translation_weight) * league_attack_mean,
        translation_weight * lower_division_defense
        + (1 - translation_weight) * league_defense_mean,
        effective_matches=translation_weight * 8.0,
        provenance="translated_lower_division_prior",
    )


def transfer_strength_prior(
    *,
    league_attack_mean: float,
    league_defense_mean: float,
    lower_division_attack: float | None = None,
    lower_division_defense: float | None = None,
    lower_division_weight: float = 0.0,
    country_coefficient_adjustment: float | None = None,
    squad_value_adjustment: float | None = None,
) -> StrengthPrior:
    """Blend available transfer evidence while retaining explicit provenance."""
    if not 0 <= lower_division_weight <= 1:
        raise ValueError("lower_division_weight must be in [0, 1]")
    attack, defense = float(league_attack_mean), float(league_defense_mean)
    evidence = []
    effective = 0.0
    if lower_division_attack is not None and lower_division_defense is not None:
        attack = (
            lower_division_weight * lower_division_attack
            + (1 - lower_division_weight) * attack
        )
        defense = (
            lower_division_weight * lower_division_defense
            + (1 - lower_division_weight) * defense
        )
        evidence.append("translated_lower_division")
        effective += 8.0 * lower_division_weight
    if country_coefficient_adjustment is not None:
        attack += float(country_coefficient_adjustment)
        defense -= float(country_coefficient_adjustment)
        evidence.append("country_coefficient")
        effective += 2.0
    if squad_value_adjustment is not None:
        adjustment = max(-0.35, min(0.35, float(squad_value_adjustment)))
        attack += adjustment
        defense -= adjustment
        evidence.append("squad_value")
        effective += 3.0
    return StrengthPrior(
        attack_mean=attack,
        defense_mean=defense,
        effective_matches=effective,
        provenance="+".join(evidence) if evidence else "conservative_league_prior",
    )


def cold_start_badge(effective_matches: float, *, full_history_threshold: float = 10.0) -> str:
    if effective_matches < 0 or full_history_threshold <= 0:
        raise ValueError("invalid cold-start thresholds")
    if effective_matches == 0:
        return "league_prior"
    if effective_matches < full_history_threshold:
        return "promoted_or_partial_prior"
    return "full_history"
