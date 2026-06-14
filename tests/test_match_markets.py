import pandas as pd
import pytest

from wc2026_model.markets.match_odds import (
    compare_match_probabilities,
    prepare_match_market_snapshot,
)


def test_prepare_match_market_snapshot_adds_raw_and_no_vig_probabilities() -> None:
    snapshot = pd.DataFrame(
        [
            {
                "match_date": "2026-06-15",
                "home_team": "France",
                "away_team": "Senegal",
                "home_decimal_odds": 1.95,
                "draw_decimal_odds": 3.4,
                "away_decimal_odds": 4.6,
            }
        ]
    )

    prepared = prepare_match_market_snapshot(snapshot)

    assert prepared.loc[0, "overround"] > 1.0
    assert (
        prepared.loc[0, "home_no_vig_probability"]
        + prepared.loc[0, "draw_no_vig_probability"]
        + prepared.loc[0, "away_no_vig_probability"]
    ) == pytest.approx(1.0)


def test_compare_match_probabilities_computes_edges_and_expected_value() -> None:
    model_probabilities = pd.DataFrame(
        [
            {
                "match_date": "2026-06-15",
                "home_team": "France",
                "away_team": "Senegal",
                "home_win_probability": 0.56,
                "draw_probability": 0.24,
                "away_win_probability": 0.20,
            }
        ]
    )
    snapshot = pd.DataFrame(
        [
            {
                "match_date": "2026-06-15",
                "home_team": "France",
                "away_team": "Senegal",
                "home_decimal_odds": 1.95,
                "draw_decimal_odds": 3.4,
                "away_decimal_odds": 4.6,
            }
        ]
    )

    comparison = compare_match_probabilities(model_probabilities, snapshot)

    assert comparison.loc[0, "home_edge_vs_no_vig"] > 0.0
    assert comparison.loc[0, "home_expected_value"] == pytest.approx((0.56 * 1.95) - 1.0)
