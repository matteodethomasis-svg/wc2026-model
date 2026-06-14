import numpy as np
import pytest

from wc2026_model.models import (
    BlendedMatchModel,
    blend_three_way_probabilities,
    reweight_score_matrix_to_outcomes,
    three_way_probabilities_from_score_matrix,
)
from wc2026_model.types import ThreeWayProbabilities


def test_blend_three_way_probabilities_interpolates_linearly() -> None:
    base = ThreeWayProbabilities(home=0.50, draw=0.30, away=0.20)
    overlay = ThreeWayProbabilities(home=0.60, draw=0.25, away=0.15)

    blended = blend_three_way_probabilities(base, overlay, alpha_on_base=0.75)

    assert blended == ThreeWayProbabilities(
        home=pytest.approx(0.525),
        draw=pytest.approx(0.2875),
        away=pytest.approx(0.1875),
    )


def test_reweight_score_matrix_to_outcomes_matches_target_probabilities() -> None:
    score_matrix = np.array(
        [
            [0.20, 0.10],
            [0.15, 0.55],
        ],
        dtype=float,
    )
    target = ThreeWayProbabilities(home=0.30, draw=0.50, away=0.20)

    reweighted = reweight_score_matrix_to_outcomes(
        score_matrix,
        target_probabilities=target,
    )
    probabilities = three_way_probabilities_from_score_matrix(reweighted)

    assert reweighted.sum() == pytest.approx(1.0)
    assert probabilities.home == pytest.approx(target.home)
    assert probabilities.draw == pytest.approx(target.draw)
    assert probabilities.away == pytest.approx(target.away)


class _BaseModel:
    teams = ["France", "Senegal"]

    def predict_expected_goals(self, *args, **kwargs):
        return 1.4, 0.8

    def predict_outcome_probabilities(self, *args, **kwargs):
        return ThreeWayProbabilities(home=0.50, draw=0.30, away=0.20)

    def predict_score_matrix(self, *args, **kwargs):
        return np.array(
            [
                [0.18, 0.10, 0.02],
                [0.16, 0.24, 0.05],
                [0.12, 0.07, 0.06],
            ],
            dtype=float,
        )


class _OverlayModel:
    def predict_proba(self, row):
        assert hasattr(row, "elo_diff_pre")
        assert hasattr(row, "neutral")
        return ThreeWayProbabilities(home=0.60, draw=0.25, away=0.15)


def test_blended_match_model_blends_outcomes_and_reweights_matrix() -> None:
    model = BlendedMatchModel(
        base_model=_BaseModel(),
        overlay_model=_OverlayModel(),
        alpha_on_base=0.75,
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

    assert probabilities.home == pytest.approx(0.525)
    assert probabilities.draw == pytest.approx(0.2875)
    assert probabilities.away == pytest.approx(0.1875)
    assert matrix_probabilities.home == pytest.approx(probabilities.home)
    assert matrix_probabilities.draw == pytest.approx(probabilities.draw)
    assert matrix_probabilities.away == pytest.approx(probabilities.away)
