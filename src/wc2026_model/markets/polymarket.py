from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from wc2026_model.data import canonicalize_team_name

POLYMARKET_GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
DEFAULT_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


@dataclass(frozen=True)
class PolymarketEventQuery:
    tag_slug: str = "2026-fifa-world-cup"
    event_slug: str = "world-cup-winner"
    limit: int = 500
    closed: bool = False


def fetch_polymarket_events(query: PolymarketEventQuery = PolymarketEventQuery()) -> list[dict[str, Any]]:
    query_string = urlencode(
        {
            "limit": query.limit,
            "closed": str(query.closed).lower(),
            "tag_slug": query.tag_slug,
        }
    )
    request = Request(
        f"{POLYMARKET_GAMMA_EVENTS_URL}?{query_string}",
        headers=DEFAULT_HTTP_HEADERS,
    )
    with urlopen(request) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError("Polymarket events response must be a JSON list.")
    return payload


def find_polymarket_event(
    events: list[dict[str, Any]],
    *,
    event_slug: str,
) -> dict[str, Any]:
    for event in events:
        if str(event.get("slug", "")).strip() == event_slug:
            return event
    raise KeyError(f"Unable to find Polymarket event slug '{event_slug}'.")


def extract_world_cup_winner_market_frame(event: dict[str, Any]) -> pd.DataFrame:
    markets = event.get("markets")
    if not isinstance(markets, list):
        raise ValueError("Polymarket event is missing the markets list.")

    rows: list[dict[str, Any]] = []
    for market in markets:
        if not isinstance(market, dict):
            continue

        market_team = str(market.get("groupItemTitle", "")).strip()
        if not market_team:
            continue

        outcomes = _coerce_list(market.get("outcomes"))
        outcome_prices = _coerce_float_list(_coerce_list(market.get("outcomePrices")))
        outcome_lookup = {
            str(outcome).strip(): price
            for outcome, price in zip(outcomes, outcome_prices, strict=False)
        }

        yes_price = _first_non_null(
            outcome_lookup.get("Yes"),
            outcome_prices[0] if outcome_prices else None,
        )
        no_price = _first_non_null(
            outcome_lookup.get("No"),
            outcome_prices[1] if len(outcome_prices) > 1 else None,
        )
        last_trade_price = _coerce_float(market.get("lastTradePrice"))

        rows.append(
            {
                "team": canonicalize_team_name(market_team),
                "market_team": market_team,
                "event_slug": str(event.get("slug", "")),
                "market_slug": str(market.get("slug", "")),
                "yes_price": yes_price,
                "no_price": no_price,
                "last_trade_price": last_trade_price,
                "best_bid": _coerce_float(market.get("bestBid")),
                "best_ask": _coerce_float(market.get("bestAsk")),
                "volume": _coerce_float(market.get("volume")),
                "liquidity": _coerce_float(market.get("liquidity")),
                "market_probability": _first_non_null(yes_price, last_trade_price),
            }
        )

    output = pd.DataFrame.from_records(rows)
    if output.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "market_team",
                "event_slug",
                "market_slug",
                "yes_price",
                "no_price",
                "last_trade_price",
                "best_bid",
                "best_ask",
                "volume",
                "liquidity",
                "market_probability",
            ]
        )

    return output.sort_values(
        ["market_probability", "team"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def compare_world_cup_winner_probabilities(
    model_probabilities: pd.DataFrame,
    market_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    required_model_columns = {"team", "champion_probability"}
    missing_model = required_model_columns.difference(model_probabilities.columns)
    if missing_model:
        missing = ", ".join(sorted(missing_model))
        raise ValueError(f"Model probabilities are missing columns: {missing}")

    required_market_columns = {"team", "market_probability"}
    missing_market = required_market_columns.difference(market_probabilities.columns)
    if missing_market:
        missing = ", ".join(sorted(missing_market))
        raise ValueError(f"Market probabilities are missing columns: {missing}")

    model_frame = model_probabilities.copy()
    market_frame = market_probabilities.copy()
    model_frame["team"] = model_frame["team"].astype(str).map(canonicalize_team_name)
    market_frame["team"] = market_frame["team"].astype(str).map(canonicalize_team_name)

    merged = model_frame.merge(
        market_frame,
        on="team",
        how="inner",
        suffixes=("_model", "_market"),
    ).copy()
    merged["edge_vs_market"] = merged["champion_probability"] - merged["market_probability"]
    merged["model_fair_decimal_odds"] = merged["champion_probability"].map(_probability_to_decimal_odds)
    merged["market_fair_decimal_odds"] = merged["market_probability"].map(_probability_to_decimal_odds)
    merged["expected_value_per_yes_share"] = merged["edge_vs_market"]
    return merged.sort_values(
        ["edge_vs_market", "champion_probability", "team"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
        return [value]
    return [value]


def _coerce_float_list(values: list[Any]) -> list[float | None]:
    return [_coerce_float(value) for value in values]


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_non_null(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _probability_to_decimal_odds(probability: float) -> float | None:
    if probability <= 0.0:
        return None
    return 1.0 / probability
