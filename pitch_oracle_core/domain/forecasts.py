"""Coherent match forecast and goal-market projections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import numpy as np

from .probability_grid import ProbabilityGrid


def markets_from_score_matrix(matrix: np.ndarray) -> dict[str, float | str]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("matrix must be finite, non-negative, and two-dimensional")
    if matrix.sum() <= 0:
        raise ValueError("matrix must contain positive probability mass")
    matrix = matrix / matrix.sum()
    home_goals = np.arange(matrix.shape[0])[:, None]
    away_goals = np.arange(matrix.shape[1])[None, :]
    total = home_goals + away_goals
    mode = np.unravel_index(int(matrix.argmax()), matrix.shape)
    home_win = float(matrix[home_goals > away_goals].sum())
    draw = float(matrix[home_goals == away_goals].sum())
    away_win = float(matrix[home_goals < away_goals].sum())
    output: dict[str, float | str] = {
        "p_home": home_win,
        "p_draw": draw,
        "p_away": away_win,
        "p_home_or_draw": home_win + draw,
        "p_away_or_draw": away_win + draw,
        "p_home_or_away": home_win + away_win,
        "p_btts_yes": float(matrix[1:, 1:].sum()),
        "p_btts_no": float(1.0 - matrix[1:, 1:].sum()),
        "p_home_clean_sheet": float(matrix[:, 0].sum()),
        "p_away_clean_sheet": float(matrix[0, :].sum()),
        "p_home_win_to_nil": float(matrix[1:, 0].sum()),
        "p_away_win_to_nil": float(matrix[0, 1:].sum()),
        "most_likely_score": f"{mode[0]}-{mode[1]}",
    }
    if home_win > 0:
        output["p_home_draw_no_bet"] = home_win / (home_win + away_win)
    if away_win > 0:
        output["p_away_draw_no_bet"] = away_win / (home_win + away_win)
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        suffix = str(line).replace(".", "_")
        output[f"p_over_{suffix}"] = float(matrix[total > line].sum())
        output[f"p_under_{suffix}"] = float(matrix[total < line].sum())
        output[f"p_home_team_over_{suffix}"] = float(
            matrix[home_goals[:, 0] > line, :].sum()
        )
        output[f"p_away_team_over_{suffix}"] = float(
            matrix[:, away_goals[0, :] > line].sum()
        )
    return output


@dataclass(frozen=True)
class MatchForecast:
    fixture_id: str
    issued_at: datetime
    model_id: str
    score_matrix: np.ndarray | ProbabilityGrid
    home_history_n: int
    away_history_n: int
    prior_weight: float
    entity_resolution_status: str

    def __post_init__(self) -> None:
        if self.issued_at.tzinfo is None:
            raise ValueError("issued_at must be timezone-aware")
        matrix = self.score_matrix.mass if isinstance(self.score_matrix, ProbabilityGrid) else self.score_matrix
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("score_matrix must be finite, non-negative, and 2D")
        total = matrix.sum()
        if isinstance(self.score_matrix, ProbabilityGrid):
            if self.score_matrix.tail_mass > 1e-8:
                raise ValueError("forecast point markets require a negligible score tail")
        elif not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError("score_matrix must sum to one")
        if min(self.home_history_n, self.away_history_n) < 0:
            raise ValueError("history counts cannot be negative")
        if not 0 <= self.prior_weight <= 1:
            raise ValueError("prior_weight must be in [0, 1]")

    @property
    def matrix(self) -> np.ndarray:
        values = self.score_matrix.mass if isinstance(self.score_matrix, ProbabilityGrid) else self.score_matrix
        values = np.asarray(values, dtype=float)
        return values / values.sum()

    @property
    def one_x_two(self) -> tuple[float, float, float]:
        markets = markets_from_score_matrix(self.matrix)
        return float(markets["p_home"]), float(markets["p_draw"]), float(markets["p_away"])

    @property
    def expected_goals(self) -> tuple[float, float]:
        home_goals = np.arange(self.matrix.shape[0])[:, None]
        away_goals = np.arange(self.matrix.shape[1])[None, :]
        return (
            float((self.matrix * home_goals).sum()),
            float((self.matrix * away_goals).sum()),
        )

    def total_over(self, line: float) -> float:
        if line < 0 or line % 1 != 0.5:
            raise ValueError("Only non-negative half-goal lines are supported")
        home_goals = np.arange(self.matrix.shape[0])[:, None]
        away_goals = np.arange(self.matrix.shape[1])[None, :]
        return float(self.matrix[(home_goals + away_goals) > line].sum())

    @property
    def btts(self) -> float:
        return float(self.matrix[1:, 1:].sum())

    @property
    def cold_start(self) -> str:
        if self.entity_resolution_status == "new_team_prior":
            return "promoted_prior"
        if min(self.home_history_n, self.away_history_n) == 0:
            return "league_prior"
        if min(self.home_history_n, self.away_history_n) < 8:
            return "partial_history"
        return "full"
