import pytest

from wc2026_model.markets.implied import (
    decimal_odds_to_implied_probabilities,
    expected_value_from_decimal_odds,
    find_positive_edges,
    remove_overround_multiplicative,
)
from wc2026_model.types import ThreeWayProbabilities


def test_decimal_odds_to_implied_probabilities() -> None:
    probabilities = decimal_odds_to_implied_probabilities(2.0, 4.0, 4.0)
    assert probabilities.home == 0.5
    assert probabilities.draw == 0.25
    assert probabilities.away == 0.25


def test_remove_overround_multiplicative_normalizes_to_one() -> None:
    fair = remove_overround_multiplicative(
        ThreeWayProbabilities(home=0.52, draw=0.27, away=0.26)
    )
    total = fair.home + fair.draw + fair.away
    assert abs(total - 1.0) < 1e-12


def test_find_positive_edges_only_returns_positive_differences() -> None:
    edges = find_positive_edges(
        model_probabilities=ThreeWayProbabilities(home=0.50, draw=0.25, away=0.25),
        fair_market_probabilities=ThreeWayProbabilities(home=0.45, draw=0.28, away=0.27),
    )
    assert edges["home"] == pytest.approx(0.05)
    assert set(edges) == {"home"}


def test_expected_value_from_decimal_odds() -> None:
    ev = expected_value_from_decimal_odds(model_probability=0.55, decimal_odds=2.10)
    assert abs(ev - 0.155) < 1e-12
