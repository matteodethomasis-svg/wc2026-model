from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from wc2026_model.types import ThreeWayProbabilities

if TYPE_CHECKING:
    from wc2026_model.evaluation.benchmarking import EloMultinomialBenchmark


def blend_three_way_probabilities(
    base: ThreeWayProbabilities,
    overlay: ThreeWayProbabilities,
    *,
    alpha_on_base: float,
) -> ThreeWayProbabilities:
    if not 0.0 <= alpha_on_base <= 1.0:
        raise ValueError(f"alpha_on_base must lie in [0, 1], got {alpha_on_base}.")
    alpha_on_overlay = 1.0 - alpha_on_base
    return ThreeWayProbabilities(
        home=(alpha_on_base * base.home) + (alpha_on_overlay * overlay.home),
        draw=(alpha_on_base * base.draw) + (alpha_on_overlay * overlay.draw),
        away=(alpha_on_base * base.away) + (alpha_on_overlay * overlay.away),
    )


def log_pool_three_way_probabilities(
    base: ThreeWayProbabilities,
    overlay: ThreeWayProbabilities,
    *,
    alpha_on_base: float,
    temperature: float = 1.0,
    floor: float = 1e-3,
) -> ThreeWayProbabilities:
    """Log-opinion pooling: weighted GEOMETRIC mean of the two probability vectors,
    then renormalize (optionally temperature-scaled). This is the blend DrElegantia's
    ensemble uses (`_pool`): vs the arithmetic convex blend it weights AGREEMENT more
    and is sharper, which can help calibration when both models are decent. With
    temperature=1 and the same alpha it is the natural geometric counterpart of the
    linear blend; temperature>1 softens, <1 sharpens the pooled distribution."""
    if not 0.0 <= alpha_on_base <= 1.0:
        raise ValueError(f"alpha_on_base must lie in [0, 1], got {alpha_on_base}.")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    alpha_on_overlay = 1.0 - alpha_on_base

    def _pool(b: float, o: float) -> float:
        b = min(max(b, floor), 1.0)
        o = min(max(o, floor), 1.0)
        log_mix = alpha_on_base * math.log(b) + alpha_on_overlay * math.log(o)
        return math.exp(log_mix / temperature)

    home = _pool(base.home, overlay.home)
    draw = _pool(base.draw, overlay.draw)
    away = _pool(base.away, overlay.away)
    total = home + draw + away
    if total <= 0.0:
        return ThreeWayProbabilities(home=1 / 3, draw=1 / 3, away=1 / 3)
    return ThreeWayProbabilities(home=home / total, draw=draw / total, away=away / total)


def three_way_probabilities_from_score_matrix(score_matrix: np.ndarray) -> ThreeWayProbabilities:
    matrix = _normalize_score_matrix(score_matrix)
    return ThreeWayProbabilities(
        home=float(np.tril(matrix, k=-1).sum()),
        draw=float(np.trace(matrix)),
        away=float(np.triu(matrix, k=1).sum()),
    )


def reweight_score_matrix_to_outcomes(
    score_matrix: np.ndarray,
    *,
    target_probabilities: ThreeWayProbabilities,
) -> np.ndarray:
    matrix = _normalize_score_matrix(score_matrix)
    current_probabilities = three_way_probabilities_from_score_matrix(matrix)
    scaled = matrix.copy()

    _scale_outcome_region(
        scaled,
        mask=np.tril(np.ones_like(scaled, dtype=bool), k=-1),
        current_probability=current_probabilities.home,
        target_probability=target_probabilities.home,
        outcome_name="home",
    )
    _scale_outcome_region(
        scaled,
        mask=np.eye(scaled.shape[0], scaled.shape[1], dtype=bool),
        current_probability=current_probabilities.draw,
        target_probability=target_probabilities.draw,
        outcome_name="draw",
    )
    _scale_outcome_region(
        scaled,
        mask=np.triu(np.ones_like(scaled, dtype=bool), k=1),
        current_probability=current_probabilities.away,
        target_probability=target_probabilities.away,
        outcome_name="away",
    )
    return _normalize_score_matrix(scaled)


def _scale_outcome_region(
    score_matrix: np.ndarray,
    *,
    mask: np.ndarray,
    current_probability: float,
    target_probability: float,
    outcome_name: str,
) -> None:
    if current_probability <= 0.0:
        if target_probability > 1e-12:
            raise ValueError(
                f"Cannot assign positive probability to {outcome_name} because the base matrix has zero mass there."
            )
        score_matrix[mask] = 0.0
        return
    scale = target_probability / current_probability
    score_matrix[mask] *= scale


def _normalize_score_matrix(score_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(score_matrix, dtype=float)
    total = float(matrix.sum())
    if total <= 0.0:
        raise ValueError("Score matrix must have strictly positive mass.")
    return matrix / total


@dataclass
class BlendedMatchModel:
    base_model: object
    overlay_model: "EloMultinomialBenchmark"
    alpha_on_base: float
    # "linear" = arithmetic convex blend (original). "log_pool" = weighted geometric
    # mean (log-opinion pooling); validated to beat linear by ~0.5% log loss on WC18+22
    # (idea from DrElegantia's ensemble). temperature only applies to log_pool.
    blend_method: str = "linear"
    blend_temperature: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.alpha_on_base) <= 1.0:
            raise ValueError(
                f"alpha_on_base must lie in [0, 1], got {self.alpha_on_base}."
            )
        if self.blend_method not in ("linear", "log_pool"):
            raise ValueError(
                f"blend_method must be 'linear' or 'log_pool', got {self.blend_method!r}."
            )
        self.teams = getattr(self.base_model, "teams", None)

    def predict_expected_goals(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool = False,
        elo_diff_pre: float = 0.0,
    ) -> tuple[float, float]:
        return self.base_model.predict_expected_goals(
            home_team,
            away_team,
            neutral_site=neutral_site,
            elo_diff_pre=elo_diff_pre,
        )

    def predict_outcome_probabilities(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool = False,
        elo_diff_pre: float = 0.0,
        max_goals: int = 10,
    ) -> ThreeWayProbabilities:
        base_probabilities = self.base_model.predict_outcome_probabilities(
            home_team,
            away_team,
            neutral_site=neutral_site,
            elo_diff_pre=elo_diff_pre,
            max_goals=max_goals,
        )
        overlay_probabilities = self.overlay_model.predict_proba(
            SimpleNamespace(
                elo_diff_pre=float(elo_diff_pre),
                neutral=bool(neutral_site),
            )
        )
        if self.blend_method == "log_pool":
            return log_pool_three_way_probabilities(
                base_probabilities,
                overlay_probabilities,
                alpha_on_base=float(self.alpha_on_base),
                temperature=float(self.blend_temperature),
            )
        return blend_three_way_probabilities(
            base_probabilities,
            overlay_probabilities,
            alpha_on_base=float(self.alpha_on_base),
        )

    def predict_score_matrix(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool = False,
        elo_diff_pre: float = 0.0,
        max_goals: int = 10,
    ) -> np.ndarray:
        base_matrix = self.base_model.predict_score_matrix(
            home_team,
            away_team,
            neutral_site=neutral_site,
            elo_diff_pre=elo_diff_pre,
            max_goals=max_goals,
        )
        blended_probabilities = self.predict_outcome_probabilities(
            home_team,
            away_team,
            neutral_site=neutral_site,
            elo_diff_pre=elo_diff_pre,
            max_goals=max_goals,
        )
        return reweight_score_matrix_to_outcomes(
            base_matrix,
            target_probabilities=blended_probabilities,
        )
