from __future__ import annotations

from wc2026_model.types import (
    OUTCOME_AWAY,
    OUTCOME_DRAW,
    OUTCOME_HOME,
    THREE_WAY_OUTCOMES,
    ThreeWayProbabilities,
)


def _validate_decimal_odds(value: float) -> None:
    if value <= 1.0:
        raise ValueError(f"Decimal odds must be greater than 1.0, got {value}.")


def decimal_odds_to_implied_probabilities(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
) -> ThreeWayProbabilities:
    _validate_decimal_odds(home_odds)
    _validate_decimal_odds(draw_odds)
    _validate_decimal_odds(away_odds)
    return ThreeWayProbabilities(
        home=1.0 / home_odds,
        draw=1.0 / draw_odds,
        away=1.0 / away_odds,
    )


def normalize_probabilities(probabilities: ThreeWayProbabilities) -> ThreeWayProbabilities:
    total = probabilities.home + probabilities.draw + probabilities.away
    if total <= 0.0:
        raise ValueError("Probability mass must be strictly positive.")
    return ThreeWayProbabilities(
        home=probabilities.home / total,
        draw=probabilities.draw / total,
        away=probabilities.away / total,
    )


def remove_overround_multiplicative(probabilities: ThreeWayProbabilities) -> ThreeWayProbabilities:
    return normalize_probabilities(probabilities)


def find_positive_edges(
    model_probabilities: ThreeWayProbabilities,
    fair_market_probabilities: ThreeWayProbabilities,
) -> dict[str, float]:
    model = model_probabilities.as_dict()
    market = fair_market_probabilities.as_dict()
    return {
        outcome: model[outcome] - market[outcome]
        for outcome in THREE_WAY_OUTCOMES
        if model[outcome] > market[outcome]
    }


def expected_value_from_decimal_odds(model_probability: float, decimal_odds: float) -> float:
    _validate_decimal_odds(decimal_odds)
    return (model_probability * decimal_odds) - 1.0


def quote_to_comparison_dict(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
) -> dict[str, float]:
    raw = decimal_odds_to_implied_probabilities(home_odds, draw_odds, away_odds)
    fair = remove_overround_multiplicative(raw)
    return {
        f"raw_{OUTCOME_HOME}": raw.home,
        f"raw_{OUTCOME_DRAW}": raw.draw,
        f"raw_{OUTCOME_AWAY}": raw.away,
        f"fair_{OUTCOME_HOME}": fair.home,
        f"fair_{OUTCOME_DRAW}": fair.draw,
        f"fair_{OUTCOME_AWAY}": fair.away,
    }
