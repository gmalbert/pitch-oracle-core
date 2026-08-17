"""Research-only pitch-control approximation over player locations."""

import numpy as np


def pitch_control_probability(
    attacking_positions: np.ndarray,
    defending_positions: np.ndarray,
    target: tuple[float, float],
    *,
    attacking_speed: float = 5.0,
    defending_speed: float = 5.0,
) -> float:
    attack = np.asarray(attacking_positions, dtype=float)
    defense = np.asarray(defending_positions, dtype=float)
    if attack.ndim != 2 or defense.ndim != 2 or attack.shape[1] != 2 or defense.shape[1] != 2:
        raise ValueError("positions must have shape (players, 2)")
    if attacking_speed <= 0 or defending_speed <= 0:
        raise ValueError("player speeds must be positive")
    target_array = np.asarray(target, dtype=float)
    attack_time = np.linalg.norm(attack - target_array, axis=1).min() / attacking_speed
    defense_time = np.linalg.norm(defense - target_array, axis=1).min() / defending_speed
    return float(1 / (1 + np.exp((attack_time - defense_time) * 4)))
