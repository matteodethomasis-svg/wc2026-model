from __future__ import annotations

import numpy as np
import pandas as pd

from wc2026_model.types import (
    OUTCOME_AWAY,
    OUTCOME_DRAW,
    OUTCOME_HOME,
    THREE_WAY_OUTCOMES,
    ThreeWayProbabilities,
)


def expected_calibration_error_three_way(
    prediction_frame: pd.DataFrame,
    *,
    actual_outcome_column: str = "actual_outcome",
    probability_column_map: dict[str, str] | None = None,
    bins: int = 10,
) -> float:
    if prediction_frame.empty:
        return 0.0

    probability_column_map = probability_column_map or {
        OUTCOME_HOME: "pred_home",
        OUTCOME_DRAW: "pred_draw",
        OUTCOME_AWAY: "pred_away",
    }
    ece_values = []
    for outcome in THREE_WAY_OUTCOMES:
        probability_column = probability_column_map[outcome]
        probabilities = prediction_frame[probability_column].to_numpy(dtype=float)
        actuals = (
            prediction_frame[actual_outcome_column].astype(str).to_numpy() == outcome
        ).astype(float)
        ece_values.append(_binary_expected_calibration_error(probabilities, actuals, bins=bins))
    return float(np.mean(ece_values))


def probabilities_to_row(probabilities: ThreeWayProbabilities) -> dict[str, float]:
    return {
        "pred_home": probabilities.home,
        "pred_draw": probabilities.draw,
        "pred_away": probabilities.away,
    }


def power_calibrate_three_way(
    probabilities: ThreeWayProbabilities,
    *,
    gamma_home: float = 1.0,
    gamma_draw: float = 1.0,
    gamma_away: float = 1.0,
    epsilon: float = 1e-12,
) -> ThreeWayProbabilities:
    gammas = {
        OUTCOME_HOME: float(gamma_home),
        OUTCOME_DRAW: float(gamma_draw),
        OUTCOME_AWAY: float(gamma_away),
    }
    if any(gamma <= 0.0 for gamma in gammas.values()):
        raise ValueError(f"All calibration gammas must be positive, got {gammas}.")

    raw_probability_map = {
        OUTCOME_HOME: float(probabilities.home),
        OUTCOME_DRAW: float(probabilities.draw),
        OUTCOME_AWAY: float(probabilities.away),
    }
    adjusted_probability_map = {
        outcome: max(raw_probability_map[outcome], epsilon) ** gammas[outcome]
        for outcome in THREE_WAY_OUTCOMES
    }
    total = sum(adjusted_probability_map.values())
    if total <= 0.0:
        raise ValueError("Adjusted probability mass must be strictly positive.")

    return ThreeWayProbabilities(
        home=adjusted_probability_map[OUTCOME_HOME] / total,
        draw=adjusted_probability_map[OUTCOME_DRAW] / total,
        away=adjusted_probability_map[OUTCOME_AWAY] / total,
    )


def power_calibrate_prediction_frame(
    prediction_frame: pd.DataFrame,
    *,
    gamma_home: float = 1.0,
    gamma_draw: float = 1.0,
    gamma_away: float = 1.0,
    probability_column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    probability_column_map = probability_column_map or {
        OUTCOME_HOME: "pred_home",
        OUTCOME_DRAW: "pred_draw",
        OUTCOME_AWAY: "pred_away",
    }
    required_columns = set(probability_column_map.values())
    missing_columns = required_columns.difference(prediction_frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Prediction frame is missing required probability columns: {missing}")

    calibrated = prediction_frame.copy()
    calibrated_probabilities = calibrated.apply(
        lambda row: power_calibrate_three_way(
            ThreeWayProbabilities(
                home=float(row[probability_column_map[OUTCOME_HOME]]),
                draw=float(row[probability_column_map[OUTCOME_DRAW]]),
                away=float(row[probability_column_map[OUTCOME_AWAY]]),
            ),
            gamma_home=gamma_home,
            gamma_draw=gamma_draw,
            gamma_away=gamma_away,
        ),
        axis=1,
    )
    calibrated[probability_column_map[OUTCOME_HOME]] = calibrated_probabilities.map(
        lambda probabilities: probabilities.home
    )
    calibrated[probability_column_map[OUTCOME_DRAW]] = calibrated_probabilities.map(
        lambda probabilities: probabilities.draw
    )
    calibrated[probability_column_map[OUTCOME_AWAY]] = calibrated_probabilities.map(
        lambda probabilities: probabilities.away
    )
    return calibrated


def _binary_expected_calibration_error(
    probabilities: np.ndarray,
    actuals: np.ndarray,
    *,
    bins: int,
) -> float:
    clipped_probabilities = np.clip(probabilities, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    observation_count = len(clipped_probabilities)
    for bin_index in range(bins):
        left = bin_edges[bin_index]
        right = bin_edges[bin_index + 1]
        if bin_index == bins - 1:
            mask = (clipped_probabilities >= left) & (clipped_probabilities <= right)
        else:
            mask = (clipped_probabilities >= left) & (clipped_probabilities < right)
        if not np.any(mask):
            continue
        bin_probabilities = clipped_probabilities[mask]
        bin_actuals = actuals[mask]
        ece += (
            len(bin_probabilities) / observation_count
        ) * abs(float(bin_probabilities.mean()) - float(bin_actuals.mean()))
    return float(ece)
