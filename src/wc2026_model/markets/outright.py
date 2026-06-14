from __future__ import annotations

import math

import pandas as pd

from wc2026_model.data import canonicalize_team_name


def fractional_odds_to_decimal(odds: str) -> float:
    normalized = str(odds).strip()
    if "/" not in normalized:
        raise ValueError(f"Fractional odds must contain '/', got {odds!r}.")

    numerator_text, denominator_text = normalized.split("/", maxsplit=1)
    numerator = float(numerator_text)
    denominator = float(denominator_text)
    if denominator <= 0.0:
        raise ValueError(f"Fractional odds denominator must be positive, got {odds!r}.")
    if numerator < 0.0:
        raise ValueError(f"Fractional odds numerator must be non-negative, got {odds!r}.")
    return 1.0 + (numerator / denominator)


def decimal_odds_to_implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError(f"Decimal odds must be greater than 1.0, got {decimal_odds}.")
    return 1.0 / decimal_odds


def prepare_outright_market_snapshot(
    market_snapshot: pd.DataFrame,
    *,
    odds_column: str | None = None,
    odds_format: str = "fractional",
) -> pd.DataFrame:
    if odds_column is None:
        odds_column = "odds_fractional" if odds_format == "fractional" else "decimal_odds"
    required_columns = {"team", odds_column}
    missing_columns = required_columns.difference(market_snapshot.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Market snapshot is missing columns: {missing}")

    prepared = market_snapshot.copy()
    prepared["team"] = prepared["team"].astype(str).map(canonicalize_team_name)
    if odds_format == "fractional":
        prepared["decimal_odds"] = prepared[odds_column].map(fractional_odds_to_decimal)
    elif odds_format == "decimal":
        prepared["decimal_odds"] = prepared[odds_column].astype(float)
    else:
        raise ValueError(f"Unsupported odds format: {odds_format!r}.")
    prepared["raw_implied_probability"] = prepared["decimal_odds"].map(
        decimal_odds_to_implied_probability
    )
    total_raw_probability = float(prepared["raw_implied_probability"].sum())
    prepared["snapshot_share_probability"] = prepared["raw_implied_probability"] / total_raw_probability
    return prepared.sort_values(
        ["raw_implied_probability", "team"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def compare_outright_probabilities(
    model_probabilities: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    *,
    model_probability_column: str = "champion_probability",
    market_odds_column: str | None = None,
    market_odds_format: str = "fractional",
) -> pd.DataFrame:
    required_model_columns = {"team", model_probability_column}
    missing_model_columns = required_model_columns.difference(model_probabilities.columns)
    if missing_model_columns:
        missing = ", ".join(sorted(missing_model_columns))
        raise ValueError(f"Model probabilities are missing columns: {missing}")

    prepared_market = prepare_outright_market_snapshot(
        market_snapshot,
        odds_column=market_odds_column,
        odds_format=market_odds_format,
    )
    prepared_model = model_probabilities.copy()
    prepared_model["team"] = prepared_model["team"].astype(str).map(canonicalize_team_name)

    comparison = prepared_model.merge(prepared_market, on="team", how="inner").copy()
    comparison["edge_vs_bookmaker_raw"] = (
        comparison[model_probability_column] - comparison["raw_implied_probability"]
    )
    comparison["edge_vs_bookmaker_snapshot_share"] = (
        comparison[model_probability_column] - comparison["snapshot_share_probability"]
    )
    comparison["model_fair_decimal_odds"] = comparison[model_probability_column].map(
        _probability_to_decimal_odds
    )
    return comparison.sort_values(
        ["edge_vs_bookmaker_raw", model_probability_column, "team"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _probability_to_decimal_odds(probability: float) -> float | None:
    if probability <= 0.0 or not math.isfinite(probability):
        return None
    return 1.0 / probability
