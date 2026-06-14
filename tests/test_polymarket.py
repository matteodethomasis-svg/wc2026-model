import pandas as pd
import pytest

from wc2026_model.markets.polymarket import (
    compare_world_cup_winner_probabilities,
    extract_world_cup_winner_market_frame,
)


def test_extract_world_cup_winner_market_frame_parses_team_prices() -> None:
    event = {
        "slug": "world-cup-winner",
        "markets": [
            {
                "groupItemTitle": "Côte d'Ivoire",
                "slug": "ivory-coast-to-win-world-cup",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.031", "0.969"]',
                "bestBid": "0.030",
                "bestAsk": "0.032",
                "lastTradePrice": "0.031",
                "volume": "12345.67",
                "liquidity": "4567.89",
            }
        ],
    }

    market_frame = extract_world_cup_winner_market_frame(event)

    assert list(market_frame["team"]) == ["Ivory Coast"]
    assert market_frame.loc[0, "market_probability"] == 0.031
    assert market_frame.loc[0, "best_bid"] == 0.03
    assert market_frame.loc[0, "best_ask"] == 0.032


def test_compare_world_cup_winner_probabilities_computes_edges() -> None:
    model = pd.DataFrame(
        [
            {"team": "England", "champion_probability": 0.14},
            {"team": "France", "champion_probability": 0.11},
        ]
    )
    market = pd.DataFrame(
        [
            {"team": "England", "market_probability": 0.11},
            {"team": "France", "market_probability": 0.13},
        ]
    )

    comparison = compare_world_cup_winner_probabilities(model, market)

    assert comparison.loc[0, "team"] == "England"
    assert comparison.loc[0, "edge_vs_market"] == pytest.approx(0.03)
    assert comparison.loc[1, "team"] == "France"
    assert comparison.loc[1, "edge_vs_market"] == pytest.approx(-0.02)
