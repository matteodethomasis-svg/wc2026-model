from wc2026_model.evaluation.scoring import (
    brier_score_three_way,
    log_loss_three_way,
    ranked_probability_score,
)
from wc2026_model.types import ThreeWayProbabilities


def test_log_loss_three_way() -> None:
    probs = ThreeWayProbabilities(home=0.6, draw=0.25, away=0.15)
    loss = log_loss_three_way(probs, "home")
    assert 0.50 < loss < 0.52


def test_brier_score_three_way() -> None:
    probs = ThreeWayProbabilities(home=0.7, draw=0.2, away=0.1)
    score = brier_score_three_way(probs, "home")
    assert abs(score - 0.14) < 1e-12


def test_ranked_probability_score() -> None:
    probs = ThreeWayProbabilities(home=0.5, draw=0.3, away=0.2)
    score = ranked_probability_score(probs, "draw")
    assert abs(score - 0.145) < 1e-12
