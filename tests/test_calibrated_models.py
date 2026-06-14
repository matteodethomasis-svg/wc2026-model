import pytest

from wc2026_model.models import CalibratedMatchModel, power_calibrate_probabilities
from wc2026_model.models.hybrid import three_way_probabilities_from_score_matrix
from wc2026_model.types import ThreeWayProbabilities


class _BaseModel:
    teams = ["France", "Senegal"]

    def predict_expected_goals(self, *args, **kwargs):
        return 1.5, 0.8

    def predict_outcome_probabilities(self, *args, **kwargs):
        return ThreeWayProbabilities(home=0.50, draw=0.30, away=0.20)

    def predict_score_matrix(self, *args, **kwargs):
        return [
            [0.18, 0.10, 0.02],
            [0.16, 0.24, 0.05],
            [0.12, 0.07, 0.06],
        ]


def test_power_calibrate_probabilities_adjusts_mass() -> None:
    probabilities = ThreeWayProbabilities(home=0.50, draw=0.30, away=0.20)

    calibrated = power_calibrate_probabilities(
        probabilities,
        gamma_home=0.9,
        gamma_draw=1.1,
        gamma_away=1.0,
    )

    assert calibrated.home + calibrated.draw + calibrated.away == pytest.approx(1.0)
    assert calibrated.home > probabilities.home
    assert calibrated.draw < probabilities.draw


def test_calibrated_match_model_reweights_outcomes_and_score_matrix() -> None:
    model = CalibratedMatchModel(
        base_model=_BaseModel(),
        gamma_home=0.9,
        gamma_draw=1.1,
        gamma_away=1.0,
    )

    probabilities = model.predict_outcome_probabilities(
        "France",
        "Senegal",
        neutral_site=True,
        elo_diff_pre=120.0,
        max_goals=10,
    )
    score_matrix = model.predict_score_matrix(
        "France",
        "Senegal",
        neutral_site=True,
        elo_diff_pre=120.0,
        max_goals=10,
    )
    matrix_probabilities = three_way_probabilities_from_score_matrix(score_matrix)

    assert probabilities.home > 0.50
    assert probabilities.draw < 0.30
    assert matrix_probabilities.home == pytest.approx(probabilities.home)
    assert matrix_probabilities.draw == pytest.approx(probabilities.draw)
    assert matrix_probabilities.away == pytest.approx(probabilities.away)
