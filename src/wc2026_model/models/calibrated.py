from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from wc2026_model.models.hybrid import reweight_score_matrix_to_outcomes
from wc2026_model.types import ThreeWayProbabilities


def power_calibrate_probabilities(
    probabilities: ThreeWayProbabilities,
    *,
    gamma_home: float = 1.0,
    gamma_draw: float = 1.0,
    gamma_away: float = 1.0,
    epsilon: float = 1e-12,
) -> ThreeWayProbabilities:
    gammas = {
        "home": float(gamma_home),
        "draw": float(gamma_draw),
        "away": float(gamma_away),
    }
    if any(gamma <= 0.0 for gamma in gammas.values()):
        raise ValueError(f"All calibration gammas must be positive, got {gammas}.")

    adjusted_home = max(float(probabilities.home), epsilon) ** gammas["home"]
    adjusted_draw = max(float(probabilities.draw), epsilon) ** gammas["draw"]
    adjusted_away = max(float(probabilities.away), epsilon) ** gammas["away"]
    total = adjusted_home + adjusted_draw + adjusted_away
    if total <= 0.0:
        raise ValueError("Adjusted probability mass must be strictly positive.")

    return ThreeWayProbabilities(
        home=adjusted_home / total,
        draw=adjusted_draw / total,
        away=adjusted_away / total,
    )


@dataclass
class CalibratedMatchModel:
    base_model: object
    gamma_home: float = 1.0
    gamma_draw: float = 1.0
    gamma_away: float = 1.0
    symmetric_neutral_win_calibration: bool = True

    def __post_init__(self) -> None:
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
        gamma_home = self.gamma_home
        gamma_away = self.gamma_away
        if neutral_site and self.symmetric_neutral_win_calibration:
            neutral_win_gamma = sqrt(self.gamma_home * self.gamma_away)
            gamma_home = neutral_win_gamma
            gamma_away = neutral_win_gamma
        return power_calibrate_probabilities(
            base_probabilities,
            gamma_home=gamma_home,
            gamma_draw=self.gamma_draw,
            gamma_away=gamma_away,
        )

    def predict_score_matrix(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool = False,
        elo_diff_pre: float = 0.0,
        max_goals: int = 10,
    ):
        base_matrix = self.base_model.predict_score_matrix(
            home_team,
            away_team,
            neutral_site=neutral_site,
            elo_diff_pre=elo_diff_pre,
            max_goals=max_goals,
        )
        calibrated_probabilities = self.predict_outcome_probabilities(
            home_team,
            away_team,
            neutral_site=neutral_site,
            elo_diff_pre=elo_diff_pre,
            max_goals=max_goals,
        )
        return reweight_score_matrix_to_outcomes(
            base_matrix,
            target_probabilities=calibrated_probabilities,
        )
