from __future__ import annotations

import math

from wc2026_model.types import (
    OUTCOME_AWAY,
    OUTCOME_DRAW,
    OUTCOME_HOME,
    THREE_WAY_OUTCOMES,
    ThreeWayProbabilities,
)

_EPSILON = 1e-12


def _validate_outcome(actual_outcome: str) -> None:
    if actual_outcome not in THREE_WAY_OUTCOMES:
        raise ValueError(
            f"Unsupported outcome '{actual_outcome}'. Expected one of {THREE_WAY_OUTCOMES}."
        )


def _clamp(probability: float) -> float:
    return min(max(probability, _EPSILON), 1.0 - _EPSILON)


def log_loss_three_way(
    probabilities: ThreeWayProbabilities,
    actual_outcome: str,
) -> float:
    _validate_outcome(actual_outcome)
    prob = probabilities.as_dict()[actual_outcome]
    return -math.log(_clamp(prob))


def brier_score_three_way(
    probabilities: ThreeWayProbabilities,
    actual_outcome: str,
) -> float:
    _validate_outcome(actual_outcome)
    total = 0.0
    for outcome, probability in probabilities.as_dict().items():
        observed = 1.0 if outcome == actual_outcome else 0.0
        total += (probability - observed) ** 2
    return total


def ranked_probability_score(
    probabilities: ThreeWayProbabilities,
    actual_outcome: str,
) -> float:
    _validate_outcome(actual_outcome)
    probs = probabilities.as_dict()
    cumulative_probabilities = [
        probs[OUTCOME_HOME],
        probs[OUTCOME_HOME] + probs[OUTCOME_DRAW],
    ]

    if actual_outcome == OUTCOME_HOME:
        cumulative_observed = [1.0, 1.0]
    elif actual_outcome == OUTCOME_DRAW:
        cumulative_observed = [0.0, 1.0]
    else:
        cumulative_observed = [0.0, 0.0]

    squared_error = 0.0
    for forecast, observed in zip(cumulative_probabilities, cumulative_observed):
        squared_error += (forecast - observed) ** 2
    return squared_error / 2.0
