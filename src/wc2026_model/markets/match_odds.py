from __future__ import annotations

import math

import pandas as pd

from wc2026_model.data import canonicalize_team_name
from wc2026_model.markets.implied import (
    decimal_odds_to_implied_probabilities,
    expected_value_from_decimal_odds,
    remove_overround_multiplicative,
)


def prepare_match_market_snapshot(
    market_snapshot: pd.DataFrame,
    *,
    home_odds_column: str = "home_decimal_odds",
    draw_odds_column: str = "draw_decimal_odds",
    away_odds_column: str = "away_decimal_odds",
) -> pd.DataFrame:
    required_columns = {"home_team", "away_team", home_odds_column, draw_odds_column, away_odds_column}
    missing_columns = required_columns.difference(market_snapshot.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Match market snapshot is missing columns: {missing}")

    prepared = market_snapshot.copy()
    prepared["home_team"] = prepared["home_team"].astype(str).map(canonicalize_team_name)
    prepared["away_team"] = prepared["away_team"].astype(str).map(canonicalize_team_name)

    raw_probabilities = prepared.apply(
        lambda row: decimal_odds_to_implied_probabilities(
            float(row[home_odds_column]),
            float(row[draw_odds_column]),
            float(row[away_odds_column]),
        ),
        axis=1,
    )
    fair_probabilities = raw_probabilities.map(remove_overround_multiplicative)

    prepared["home_raw_implied_probability"] = raw_probabilities.map(lambda probs: probs.home)
    prepared["draw_raw_implied_probability"] = raw_probabilities.map(lambda probs: probs.draw)
    prepared["away_raw_implied_probability"] = raw_probabilities.map(lambda probs: probs.away)
    prepared["overround"] = (
        prepared["home_raw_implied_probability"]
        + prepared["draw_raw_implied_probability"]
        + prepared["away_raw_implied_probability"]
    )
    prepared["home_no_vig_probability"] = fair_probabilities.map(lambda probs: probs.home)
    prepared["draw_no_vig_probability"] = fair_probabilities.map(lambda probs: probs.draw)
    prepared["away_no_vig_probability"] = fair_probabilities.map(lambda probs: probs.away)
    return prepared.sort_values(
        [column for column in ("match_date", "home_team", "away_team") if column in prepared.columns],
        kind="stable",
    ).reset_index(drop=True)


def compare_match_probabilities(
    model_probabilities: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    *,
    model_home_probability_column: str = "home_win_probability",
    model_draw_probability_column: str = "draw_probability",
    model_away_probability_column: str = "away_win_probability",
) -> pd.DataFrame:
    required_model_columns = {
        "home_team",
        "away_team",
        model_home_probability_column,
        model_draw_probability_column,
        model_away_probability_column,
    }
    missing_model_columns = required_model_columns.difference(model_probabilities.columns)
    if missing_model_columns:
        missing = ", ".join(sorted(missing_model_columns))
        raise ValueError(f"Model probabilities are missing columns: {missing}")

    prepared_market = prepare_match_market_snapshot(market_snapshot)
    prepared_model = model_probabilities.copy()
    prepared_model["home_team"] = prepared_model["home_team"].astype(str).map(canonicalize_team_name)
    prepared_model["away_team"] = prepared_model["away_team"].astype(str).map(canonicalize_team_name)

    merge_keys = ["home_team", "away_team"]
    if "match_date" in prepared_market.columns and "match_date" in prepared_model.columns:
        prepared_market["match_date"] = prepared_market["match_date"].astype(str)
        prepared_model["match_date"] = prepared_model["match_date"].astype(str)
        merge_keys = ["match_date", *merge_keys]

    comparison = prepared_model.merge(prepared_market, on=merge_keys, how="inner").copy()
    comparison["home_edge_vs_raw"] = (
        comparison[model_home_probability_column] - comparison["home_raw_implied_probability"]
    )
    comparison["draw_edge_vs_raw"] = (
        comparison[model_draw_probability_column] - comparison["draw_raw_implied_probability"]
    )
    comparison["away_edge_vs_raw"] = (
        comparison[model_away_probability_column] - comparison["away_raw_implied_probability"]
    )
    comparison["home_edge_vs_no_vig"] = (
        comparison[model_home_probability_column] - comparison["home_no_vig_probability"]
    )
    comparison["draw_edge_vs_no_vig"] = (
        comparison[model_draw_probability_column] - comparison["draw_no_vig_probability"]
    )
    comparison["away_edge_vs_no_vig"] = (
        comparison[model_away_probability_column] - comparison["away_no_vig_probability"]
    )
    comparison["home_model_fair_decimal_odds"] = comparison[model_home_probability_column].map(
        _probability_to_decimal_odds
    )
    comparison["draw_model_fair_decimal_odds"] = comparison[model_draw_probability_column].map(
        _probability_to_decimal_odds
    )
    comparison["away_model_fair_decimal_odds"] = comparison[model_away_probability_column].map(
        _probability_to_decimal_odds
    )
    comparison["home_expected_value"] = comparison.apply(
        lambda row: expected_value_from_decimal_odds(
            float(row[model_home_probability_column]),
            float(row["home_decimal_odds"]),
        ),
        axis=1,
    )
    comparison["draw_expected_value"] = comparison.apply(
        lambda row: expected_value_from_decimal_odds(
            float(row[model_draw_probability_column]),
            float(row["draw_decimal_odds"]),
        ),
        axis=1,
    )
    comparison["away_expected_value"] = comparison.apply(
        lambda row: expected_value_from_decimal_odds(
            float(row[model_away_probability_column]),
            float(row["away_decimal_odds"]),
        ),
        axis=1,
    )
    comparison["best_model_edge_no_vig"] = comparison[
        ["home_edge_vs_no_vig", "draw_edge_vs_no_vig", "away_edge_vs_no_vig"]
    ].max(axis=1)
    comparison["best_model_ev"] = comparison[
        ["home_expected_value", "draw_expected_value", "away_expected_value"]
    ].max(axis=1)
    return comparison.sort_values(
        ["best_model_ev", "best_model_edge_no_vig", "home_team", "away_team"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _probability_to_decimal_odds(probability: float) -> float | None:
    if probability <= 0.0 or not math.isfinite(probability):
        return None
    return 1.0 / probability
