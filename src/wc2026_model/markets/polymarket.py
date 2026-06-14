from __future__ import annotations

import json
import re
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

# Polymarket spells a few nations differently from our fixtures' canonical names;
# normalize before canonicalize_team_name so the (date, home, away) merge lines up.
_POLYMARKET_NAME_FIXUPS = {
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "Korea Republic": "South Korea",
}


def _fix_polymarket_team(name: str) -> str:
    name = str(name).strip()
    return canonicalize_team_name(_POLYMARKET_NAME_FIXUPS.get(name, name))


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


# --------------------------------------------------------------------------- #
# Per-match 1X2 markets (slug pattern: fifwc-<home>-<away>-YYYY-MM-DD)
# --------------------------------------------------------------------------- #

# fifwc-fra-sen-2026-06-16  ->  date 2026-06-16. (Score / halftime variants carry an
# extra suffix and are skipped.)
_MATCH_SLUG_RE = re.compile(r"^fifwc-[a-z]{2,4}-[a-z]{2,4}-(\d{4}-\d{2}-\d{2})$")
_TITLE_VS_RE = re.compile(r"^(.*?)\s+vs\.?\s+(.*?)$")


def fetch_polymarket_world_cup_events(
    tag_slug: str = "fifa-world-cup", *, limit: int = 500, closed: bool = False
) -> list[dict[str, Any]]:
    """Fetch the broader World Cup event list (per-match + round/group markets live
    under tag 'fifa-world-cup', NOT '2026-fifa-world-cup' which is winner-only).

    The Gamma API caps a single response at 100 events, so we page with ``offset``
    until a short page (or ``limit``) is reached. Without this, the per-team round
    markets (R16/QF/SF) fall off the first page and silently go missing.
    """
    page_size = 100
    collected: list[dict[str, Any]] = []
    offset = 0
    while len(collected) < limit:
        query_string = urlencode(
            {
                "limit": page_size,
                "offset": offset,
                "closed": str(closed).lower(),
                "tag_slug": tag_slug,
            }
        )
        request = Request(
            f"{POLYMARKET_GAMMA_EVENTS_URL}?{query_string}", headers=DEFAULT_HTTP_HEADERS
        )
        with urlopen(request) as response:
            page = json.load(response)
        if not isinstance(page, list) or not page:
            break
        collected.extend(page)
        offset += page_size
        if len(page) < page_size:
            break
    return collected


def extract_match_odds_frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn the per-match (fifwc-*) events into a 1X2 decimal-odds snapshot ready for
    ``markets.match_odds.compare_match_probabilities``.

    Each match event has three Yes/No markets: "Will <home> win", "Will <away> win",
    and "Will <home> vs <away> end in a draw". We read each Yes price as the outcome
    probability and convert to decimal odds (1/p); the no-vig step happens downstream.
    """
    rows: list[dict[str, Any]] = []
    for event in events:
        slug = str(event.get("slug", "")).strip()
        match = _MATCH_SLUG_RE.match(slug)
        if not match:
            continue  # skip non-match, exact-score, halftime variants
        match_date = match.group(1)

        title = str(event.get("title", "")).strip()
        title_match = _TITLE_VS_RE.match(title)
        if not title_match:
            continue
        home_team = _fix_polymarket_team(title_match.group(1))
        away_team = _fix_polymarket_team(title_match.group(2))

        markets = event.get("markets")
        if not isinstance(markets, list):
            continue

        home_p = draw_p = away_p = None
        for m in markets:
            if not isinstance(m, dict):
                continue
            yes = _yes_price(m)
            if yes is None:
                continue
            git = str(m.get("groupItemTitle", "")).strip()
            question = str(m.get("question", "")).lower()
            if "draw" in git.lower() or "end in a draw" in question:
                draw_p = yes
            else:
                team = _fix_polymarket_team(git)
                if team == home_team:
                    home_p = yes
                elif team == away_team:
                    away_p = yes
        if None in (home_p, draw_p, away_p):
            continue

        rows.append(
            {
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_decimal_odds": _probability_to_decimal_odds(home_p),
                "draw_decimal_odds": _probability_to_decimal_odds(draw_p),
                "away_decimal_odds": _probability_to_decimal_odds(away_p),
                "home_market_probability": home_p,
                "draw_market_probability": draw_p,
                "away_market_probability": away_p,
                "event_slug": slug,
                "source_url": f"https://polymarket.com/event/{slug}",
                "bookmaker": "Polymarket",
                "volume": _coerce_float(event.get("volume")),
            }
        )

    columns = [
        "match_date", "home_team", "away_team",
        "home_decimal_odds", "draw_decimal_odds", "away_decimal_odds",
        "home_market_probability", "draw_market_probability", "away_market_probability",
        "event_slug", "source_url", "bookmaker", "volume",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(rows)[columns].sort_values(
        ["match_date", "home_team", "away_team"], kind="stable"
    ).reset_index(drop=True)


def _yes_price(market: dict[str, Any]) -> float | None:
    outcomes = _coerce_list(market.get("outcomes"))
    prices = _coerce_float_list(_coerce_list(market.get("outcomePrices")))
    lookup = {str(o).strip(): p for o, p in zip(outcomes, prices, strict=False)}
    return _first_non_null(
        lookup.get("Yes"),
        prices[0] if prices else None,
        _coerce_float(market.get("lastTradePrice")),
    )


# --------------------------------------------------------------------------- #
# Per-team advancement / group markets (Yes/No per nation, like the winner market)
# --------------------------------------------------------------------------- #


def extract_per_team_yes_market_frame(event: dict[str, Any]) -> pd.DataFrame:
    """Generic Yes/No-per-team extractor (reach-round, group-winner, …). Same shape as
    the winner market: each market's ``groupItemTitle`` is a nation and the Yes price is
    its probability. Returns columns: team, market_probability, volume."""
    markets = event.get("markets")
    rows: list[dict[str, Any]] = []
    if isinstance(markets, list):
        for m in markets:
            if not isinstance(m, dict):
                continue
            team = str(m.get("groupItemTitle", "")).strip()
            if not team or team.lower() in ("other", "field"):
                continue
            yes = _yes_price(m)
            if yes is None:
                continue
            rows.append(
                {
                    "team": _fix_polymarket_team(team),
                    "market_probability": yes,
                    "volume": _coerce_float(m.get("volume")),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["team", "market_probability", "volume"])
    return pd.DataFrame.from_records(rows).sort_values(
        ["market_probability", "team"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
