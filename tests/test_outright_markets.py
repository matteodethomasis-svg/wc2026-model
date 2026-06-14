import pandas as pd
import pytest

from wc2026_model.markets.outright import (
    compare_outright_probabilities,
    decimal_odds_to_implied_probability,
    fractional_odds_to_decimal,
    prepare_outright_market_snapshot,
)


def test_fractional_odds_to_decimal() -> None:
    assert fractional_odds_to_decimal("9/2") == pytest.approx(5.5)


def test_decimal_odds_to_implied_probability() -> None:
    assert decimal_odds_to_implied_probability(5.5) == pytest.approx(1.0 / 5.5)


def test_prepare_outright_market_snapshot_adds_probabilities() -> None:
    snapshot = pd.DataFrame(
        [
            {"team": "Spain", "odds_fractional": "9/2"},
            {"team": "France", "odds_fractional": "11/2"},
        ]
    )

    prepared = prepare_outright_market_snapshot(snapshot)

    assert prepared["raw_implied_probability"].sum() > 0.0
    assert prepared["snapshot_share_probability"].sum() == pytest.approx(1.0)


def test_prepare_outright_market_snapshot_supports_decimal_odds() -> None:
    snapshot = pd.DataFrame(
        [
            {"team": "France", "decimal_odds": 1.36},
            {"team": "Norway", "decimal_odds": 4.0},
        ]
    )

    prepared = prepare_outright_market_snapshot(
        snapshot,
        odds_column="decimal_odds",
        odds_format="decimal",
    )

    assert prepared.loc[0, "decimal_odds"] == pytest.approx(1.36)
    assert prepared["snapshot_share_probability"].sum() == pytest.approx(1.0)


def test_compare_outright_probabilities_joins_and_computes_edges() -> None:
    model_probabilities = pd.DataFrame(
        [
            {"team": "Spain", "champion_probability": 0.13},
            {"team": "France", "champion_probability": 0.05},
        ]
    )
    bookmaker_snapshot = pd.DataFrame(
        [
            {"team": "Spain", "odds_fractional": "9/2"},
            {"team": "France", "odds_fractional": "11/2"},
        ]
    )

    comparison = compare_outright_probabilities(model_probabilities, bookmaker_snapshot)

    assert set(comparison["team"]) == {"Spain", "France"}
    assert "edge_vs_bookmaker_raw" in comparison.columns
