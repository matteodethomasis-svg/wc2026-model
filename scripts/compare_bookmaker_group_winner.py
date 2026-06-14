from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.markets import compare_outright_probabilities


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare model group-winner probabilities against a bookmaker snapshot."
    )
    parser.add_argument(
        "--model-probabilities-input",
        default="reports/wc2026_live_simulation_probabilities.csv",
        help="CSV containing team-level simulation probabilities including group_winner_probability.",
    )
    parser.add_argument(
        "--bookmaker-snapshot-input",
        required=True,
        help="CSV containing group, team and odds_decimal columns.",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="Optional group filter, e.g. I.",
    )
    parser.add_argument(
        "--output",
        default="reports/bookmaker_group_winner_comparison.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/bookmaker_group_winner_comparison_summary.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    model_probabilities = pd.read_csv(args.model_probabilities_input)
    bookmaker_snapshot = pd.read_csv(args.bookmaker_snapshot_input)

    if args.group is not None:
        group_name = str(args.group).strip().upper()
        model_probabilities = model_probabilities.loc[
            model_probabilities["group"].astype(str).str.upper() == group_name
        ].copy()
        bookmaker_snapshot = bookmaker_snapshot.loc[
            bookmaker_snapshot["group"].astype(str).str.upper() == group_name
        ].copy()

    comparison = compare_outright_probabilities(
        model_probabilities,
        bookmaker_snapshot,
        model_probability_column="group_winner_probability",
        market_odds_column="odds_decimal",
        market_odds_format="decimal",
    )
    comparison["bookmaker_fair_decimal_odds_no_vig"] = comparison["snapshot_share_probability"].map(
        _probability_to_decimal_odds
    )
    comparison["bookmaker_fair_decimal_odds_raw"] = comparison["raw_implied_probability"].map(
        _probability_to_decimal_odds
    )

    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    summary = {
        "group": args.group,
        "team_count": int(len(comparison)),
        "top_positive_edges_vs_no_vig": comparison.head(10).loc[
            :,
            [
                "team",
                "group_winner_probability",
                "snapshot_share_probability",
                "edge_vs_bookmaker_snapshot_share",
            ],
        ].to_dict(orient="records"),
        "top_negative_edges_vs_no_vig": comparison.sort_values(
            "edge_vs_bookmaker_snapshot_share",
            ascending=True,
            kind="stable",
        )
        .head(10)
        .loc[
            :,
            [
                "team",
                "group_winner_probability",
                "snapshot_share_probability",
                "edge_vs_bookmaker_snapshot_share",
            ],
        ]
        .to_dict(orient="records"),
    }
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved group winner comparison to {output_path}")
    print(f"Saved summary to {summary_output_path}")


def _probability_to_decimal_odds(probability: float) -> float | None:
    if probability <= 0.0:
        return None
    return 1.0 / float(probability)


if __name__ == "__main__":
    main()
