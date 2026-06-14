import pandas as pd
import pytest

from wc2026_model.evaluation.calibration import (
    power_calibrate_prediction_frame,
    power_calibrate_three_way,
)
from wc2026_model.types import ThreeWayProbabilities


def test_power_calibrate_three_way_preserves_probability_mass() -> None:
    probabilities = ThreeWayProbabilities(home=0.50, draw=0.30, away=0.20)

    calibrated = power_calibrate_three_way(
        probabilities,
        gamma_home=0.9,
        gamma_draw=0.9,
        gamma_away=1.1,
    )

    assert calibrated.home + calibrated.draw + calibrated.away == pytest.approx(1.0)
    assert calibrated.home > probabilities.home


def test_power_calibrate_prediction_frame_updates_columns_in_place() -> None:
    prediction_frame = pd.DataFrame(
        [
            {
                "pred_home": 0.50,
                "pred_draw": 0.30,
                "pred_away": 0.20,
            }
        ]
    )

    calibrated = power_calibrate_prediction_frame(
        prediction_frame,
        gamma_home=0.9,
        gamma_draw=0.9,
        gamma_away=1.1,
    )

    assert calibrated.loc[0, "pred_home"] + calibrated.loc[0, "pred_draw"] + calibrated.loc[0, "pred_away"] == pytest.approx(1.0)
    assert calibrated.loc[0, "pred_home"] > prediction_frame.loc[0, "pred_home"]
