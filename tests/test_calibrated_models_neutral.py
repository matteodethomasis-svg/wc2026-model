from __future__ import annotations

from dataclasses import dataclass
from math import isclose, sqrt

from wc2026_model.models import CalibratedMatchModel
from wc2026_model.types import ThreeWayProbabilities


@dataclass
class _StubModel:
    probabilities: ThreeWayProbabilities

    def predict_expected_goals(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool = False,
        elo_diff_pre: float = 0.0,
    ) -> tuple[float, float]:
        return (1.2, 0.8)

    def predict_outcome_probabilities(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool = False,
        elo_diff_pre: float = 0.0,
        max_goals: int = 10,
    ) -> ThreeWayProbabilities:
        return self.probabilities


def test_calibrated_match_model_uses_symmetric_win_gamma_on_neutral_site() -> None:
    model = CalibratedMatchModel(
        base_model=_StubModel(
            ThreeWayProbabilities(
                home=0.50,
                draw=0.25,
                away=0.25,
            )
        ),
        gamma_home=1.05,
        gamma_draw=1.0,
        gamma_away=1.10,
    )

    probabilities = model.predict_outcome_probabilities(
        "France",
        "Senegal",
        neutral_site=True,
    )
    neutral_gamma = sqrt(1.05 * 1.10)

    expected_home = 0.50**neutral_gamma
    expected_draw = 0.25
    expected_away = 0.25**neutral_gamma
    total = expected_home + expected_draw + expected_away

    assert isclose(probabilities.home, expected_home / total)
    assert isclose(probabilities.draw, expected_draw / total)
    assert isclose(probabilities.away, expected_away / total)


def test_calibrated_match_model_keeps_asymmetric_gammas_off_neutral_site() -> None:
    model = CalibratedMatchModel(
        base_model=_StubModel(
            ThreeWayProbabilities(
                home=0.50,
                draw=0.25,
                away=0.25,
            )
        ),
        gamma_home=1.05,
        gamma_draw=1.0,
        gamma_away=1.10,
    )

    probabilities = model.predict_outcome_probabilities(
        "France",
        "Senegal",
        neutral_site=False,
    )

    expected_home = 0.50**1.05
    expected_draw = 0.25
    expected_away = 0.25**1.10
    total = expected_home + expected_draw + expected_away

    assert isclose(probabilities.home, expected_home / total)
    assert isclose(probabilities.draw, expected_draw / total)
    assert isclose(probabilities.away, expected_away / total)
