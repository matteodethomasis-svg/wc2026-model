from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.markets import compare_match_probabilities


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare model fixture probabilities against bookmaker 1X2 match odds."
    )
    parser.add_argument(
        "--model-probabilities-input",
        default="reports/wc2026_fixture_predictions.csv",
        help="CSV containing fixture-level model probabilities.",
    )
    parser.add_argument(
        "--bookmaker-snapshot-input",
        required=True,
        help="CSV containing bookmaker decimal odds with home_team, away_team and 1X2 columns.",
    )
    parser.add_argument(
        "--output",
        default="reports/bookmaker_match_odds_comparison.csv",
        help="Path to save the match-level comparison table.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/bookmaker_match_odds_comparison_summary.json",
        help="Path to save a compact summary JSON.",
    )
    return parser


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
    ordered = sorted(
        records,
        key=lambda record: (
            record["edge_vs_no_vig"],
            record["expected_value"],
            record["home_team"],
            record["away_team"],
            record["outcome"],
        ),
        reverse=largest,
    )
    return ordered[:limit]


def main() -> None:
    args = _build_parser().parse_args()

    model_probabilities = pd.read_csv(args.model_probabilities_input)
    bookmaker_snapshot = pd.read_csv(args.bookmaker_snapshot_input)
    comparison = compare_match_probabilities(model_probabilities, bookmaker_snapshot)

    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    edge_columns = [
        ("home", "home_edge_vs_no_vig", "home_no_vig_probability"),
        ("draw", "draw_edge_vs_no_vig", "draw_no_vig_probability"),
        ("away", "away_edge_vs_no_vig", "away_no_vig_probability"),
    ]
    summary = {
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
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved bookmaker match comparison to {output_path}")
    print(f"Saved summary to {summary_output_path}")


if __name__ == "__main__":
    main()
