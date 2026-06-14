from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.markets import compare_match_probabilities
from wc2026_model.models import blend_three_way_probabilities
from wc2026_model.types import ThreeWayProbabilities


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blend the current WC2026 recipe with the xG challenger using a conservative coverage-aware overlay."
    )
    parser.add_argument(
        "--current-input",
        default="reports/wc2026_fixture_predictions_expected_xi_plus_goalkeeper_recipe_2026-06-13.csv",
        help="Current WC2026 recipe predictions.",
    )
    parser.add_argument(
        "--xg-input",
        default="reports/wc2026_xg_challenger_predictions.csv",
        help="WC2026 xG challenger predictions.",
    )
    parser.add_argument(
        "--min-xg-matches-per-team",
        type=float,
        default=3.0,
        help="Coverage threshold at which the xG overlay receives full confidence scaling.",
    )
    parser.add_argument(
        "--max-xg-weight",
        type=float,
        default=0.35,
        help="Maximum weight assigned to the xG challenger before divergence shrinkage.",
    )
    parser.add_argument(
        "--delta-soft-cap",
        type=float,
        default=0.20,
        help="Probability-delta threshold above which xG weight is shrunk.",
    )
    parser.add_argument(
        "--output",
        default="reports/wc2026_xg_gated_blend_predictions.csv",
        help="Where to save the blended fixture predictions.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/wc2026_xg_gated_blend_summary.json",
        help="Where to save a compact JSON summary.",
    )
    parser.add_argument(
        "--bookmaker-snapshot-input",
        default="data/interim/bookmaker_match_odds_live_sample_2026-06-13.csv",
        help="Optional bookmaker snapshot for market comparison.",
    )
    parser.add_argument(
        "--bookmaker-comparison-output",
        default="reports/bookmaker_match_odds_xg_gated_blend_live_sample_comparison.csv",
        help="Where to save the bookmaker comparison for the gated blend.",
    )
    parser.add_argument(
        "--bookmaker-summary-output",
        default="reports/bookmaker_match_odds_xg_gated_blend_live_sample_comparison_summary.json",
        help="Where to save the bookmaker comparison summary for the gated blend.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    current = pd.read_csv(args.current_input)
    xg = pd.read_csv(args.xg_input)
    blended, summary = build_gated_xg_overlay(
        current_predictions=current,
        xg_predictions=xg,
        min_xg_matches_per_team=float(args.min_xg_matches_per_team),
        max_xg_weight=float(args.max_xg_weight),
        delta_soft_cap=float(args.delta_soft_cap),
    )

    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(output_path, index=False)
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    bookmaker_path = Path(args.bookmaker_snapshot_input)
    if bookmaker_path.exists():
        bookmaker_snapshot = pd.read_csv(bookmaker_path)
        bookmaker_comparison = compare_match_probabilities(blended, bookmaker_snapshot)
        bookmaker_summary = _build_bookmaker_summary(bookmaker_comparison)
        bookmaker_output_path = Path(args.bookmaker_comparison_output)
        bookmaker_summary_output_path = Path(args.bookmaker_summary_output)
        bookmaker_output_path.parent.mkdir(parents=True, exist_ok=True)
        bookmaker_summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        bookmaker_comparison.to_csv(bookmaker_output_path, index=False)
        bookmaker_summary_output_path.write_text(
            json.dumps(bookmaker_summary, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2))
    print(f"Saved gated blend predictions to {output_path}")
    print(f"Saved gated blend summary to {summary_output_path}")


def build_gated_xg_overlay(
    *,
    current_predictions: pd.DataFrame,
    xg_predictions: pd.DataFrame,
    min_xg_matches_per_team: float,
    max_xg_weight: float,
    delta_soft_cap: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if min_xg_matches_per_team <= 0.0:
        raise ValueError("min_xg_matches_per_team must be positive.")
    if not 0.0 <= max_xg_weight <= 1.0:
        raise ValueError("max_xg_weight must lie in [0, 1].")
    if delta_soft_cap <= 0.0:
        raise ValueError("delta_soft_cap must be positive.")

    current_required = {
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    }
    xg_required = {
        "match_id",
        "home_xg_match_count",
        "away_xg_match_count",
        "both_teams_have_xg_history",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    }
    missing_current = current_required.difference(current_predictions.columns)
    missing_xg = xg_required.difference(xg_predictions.columns)
    if missing_current:
        raise ValueError(f"Current predictions missing columns: {', '.join(sorted(missing_current))}")
    if missing_xg:
        raise ValueError(f"XG predictions missing columns: {', '.join(sorted(missing_xg))}")

    merged = current_predictions.merge(
        xg_predictions.rename(
            columns={
                "home_win_probability": "xg_home_win_probability",
                "draw_probability": "xg_draw_probability",
                "away_win_probability": "xg_away_win_probability",
                "home_fair_odds": "xg_home_fair_odds",
                "draw_fair_odds": "xg_draw_fair_odds",
                "away_fair_odds": "xg_away_fair_odds",
            }
        ),
        on=["match_id", "match_date", "home_team", "away_team"],
        how="inner",
    ).copy()

    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        current_probs = ThreeWayProbabilities(
            home=float(row.home_win_probability),
            draw=float(row.draw_probability),
            away=float(row.away_win_probability),
        )
        xg_probs = ThreeWayProbabilities(
            home=float(row.xg_home_win_probability),
            draw=float(row.xg_draw_probability),
            away=float(row.xg_away_win_probability),
        )
        raw_delta = max(
            abs(current_probs.home - xg_probs.home),
            abs(current_probs.draw - xg_probs.draw),
            abs(current_probs.away - xg_probs.away),
        )
        min_history = min(
            _coerce_float(row.home_xg_match_count) or 0.0,
            _coerce_float(row.away_xg_match_count) or 0.0,
        )
        coverage_factor = 0.0
        if bool(row.both_teams_have_xg_history):
            coverage_factor = min(min_history / float(min_xg_matches_per_team), 1.0)
        divergence_shrink = min(float(delta_soft_cap) / raw_delta, 1.0) if raw_delta > 0.0 else 1.0
        xg_weight = float(max_xg_weight) * coverage_factor * divergence_shrink
        blended_probs = blend_three_way_probabilities(
            current_probs,
            xg_probs,
            alpha_on_base=1.0 - xg_weight,
        )
        rows.append(
            {
                "match_id": row.match_id,
                "match_date": row.match_date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "neutral": bool(getattr(row, "neutral", True)),
                "home_xg_match_count": _coerce_float(row.home_xg_match_count),
                "away_xg_match_count": _coerce_float(row.away_xg_match_count),
                "both_teams_have_xg_history": bool(row.both_teams_have_xg_history),
                "xg_overlay_coverage_factor": coverage_factor,
                "xg_overlay_divergence_shrink": divergence_shrink,
                "xg_overlay_weight": xg_weight,
                "current_home_win_probability": current_probs.home,
                "current_draw_probability": current_probs.draw,
                "current_away_win_probability": current_probs.away,
                "xg_home_win_probability": xg_probs.home,
                "xg_draw_probability": xg_probs.draw,
                "xg_away_win_probability": xg_probs.away,
                "home_win_probability": blended_probs.home,
                "draw_probability": blended_probs.draw,
                "away_win_probability": blended_probs.away,
                "home_fair_odds": _probability_to_decimal_odds(blended_probs.home),
                "draw_fair_odds": _probability_to_decimal_odds(blended_probs.draw),
                "away_fair_odds": _probability_to_decimal_odds(blended_probs.away),
                "delta_vs_current_home": blended_probs.home - current_probs.home,
                "delta_vs_current_draw": blended_probs.draw - current_probs.draw,
                "delta_vs_current_away": blended_probs.away - current_probs.away,
                "max_abs_delta_vs_current": max(
                    abs(blended_probs.home - current_probs.home),
                    abs(blended_probs.draw - current_probs.draw),
                    abs(blended_probs.away - current_probs.away),
                ),
                "raw_max_abs_delta_current_vs_xg": raw_delta,
            }
        )

    blended = pd.DataFrame(rows).sort_values(
        ["match_date", "home_team", "away_team"],
        kind="stable",
    ).reset_index(drop=True)
    summary = _build_blend_summary(blended)
    return blended, summary


def _build_blend_summary(blended: pd.DataFrame) -> dict[str, object]:
    if blended.empty:
        return {"fixture_count": 0}
    weights = blended["xg_overlay_weight"].astype(float)
    return {
        "fixture_count": int(len(blended)),
        "fixtures_with_positive_xg_weight": int((weights > 0.0).sum()),
        "mean_xg_overlay_weight": float(weights.mean()),
        "median_xg_overlay_weight": float(weights.median()),
        "max_xg_overlay_weight": float(weights.max()),
        "largest_moves_vs_current_recipe": [
            {
                "match_date": row.match_date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "xg_overlay_weight": float(row.xg_overlay_weight),
                "max_abs_delta_vs_current": float(row.max_abs_delta_vs_current),
                "both_teams_have_xg_history": bool(row.both_teams_have_xg_history),
            }
            for row in blended.sort_values(
                ["max_abs_delta_vs_current", "xg_overlay_weight"],
                ascending=[False, False],
                kind="stable",
            )
            .head(10)
            .itertuples(index=False)
        ],
    }


def _build_bookmaker_summary(comparison: pd.DataFrame) -> dict[str, object]:
    edge_columns = [
        ("home", "home_edge_vs_no_vig", "home_no_vig_probability"),
        ("draw", "draw_edge_vs_no_vig", "draw_no_vig_probability"),
        ("away", "away_edge_vs_no_vig", "away_no_vig_probability"),
    ]
    return {
        "match_count": int(len(comparison)),
        "top_positive_edges_vs_no_vig": _top_edge_records(
            comparison,
            edge_columns=edge_columns,
            largest=True,
        ),
        "top_negative_edges_vs_no_vig": _top_edge_records(
            comparison,
            edge_columns=edge_columns,
            largest=False,
        ),
    }


def _top_edge_records(
    comparison: pd.DataFrame,
    *,
    edge_columns: list[tuple[str, str, str]],
    largest: bool,
    limit: int = 10,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for outcome_label, edge_column, market_probability_column in edge_columns:
        for row in comparison.itertuples(index=False):
            records.append(
                {
                    "match_date": getattr(row, "match_date", None),
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "outcome": outcome_label,
                    "edge_vs_no_vig": float(getattr(row, edge_column)),
                    "model_probability": float(
                        getattr(
                            row,
                            {
                                "home": "home_win_probability",
                                "draw": "draw_probability",
                                "away": "away_win_probability",
                            }[outcome_label],
                        )
                    ),
                    "market_probability": float(getattr(row, market_probability_column)),
                    "expected_value": float(
                        getattr(
                            row,
                            {
                                "home": "home_expected_value",
                                "draw": "draw_expected_value",
                                "away": "away_expected_value",
                            }[outcome_label],
                        )
                    ),
                }
            )
    return sorted(
        records,
        key=lambda record: (
            record["edge_vs_no_vig"],
            record["expected_value"],
            record["home_team"],
            record["away_team"],
            record["outcome"],
        ),
        reverse=largest,
    )[:limit]


def _coerce_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _probability_to_decimal_odds(probability: float) -> float | None:
    if probability <= 0.0:
        return None
    return 1.0 / probability


if __name__ == "__main__":
    main()
