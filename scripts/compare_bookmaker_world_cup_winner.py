from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.markets import compare_outright_probabilities, prepare_outright_market_snapshot


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare model World Cup outright probabilities against a bookmaker snapshot."
    )
    parser.add_argument(
        "--model-probabilities-input",
        default="reports/wc2026_live_simulation_probabilities.csv",
    )
    parser.add_argument(
        "--bookmaker-snapshot-input",
        default="data/interim/bookmaker_world_cup_winner_snapshot_2026-06-12.csv",
    )
    parser.add_argument(
        "--output",
        default="reports/bookmaker_world_cup_winner_comparison.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/bookmaker_world_cup_winner_comparison_summary.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    model_probabilities = pd.read_csv(args.model_probabilities_input)
    bookmaker_snapshot = pd.read_csv(args.bookmaker_snapshot_input)

    prepared_snapshot = prepare_outright_market_snapshot(bookmaker_snapshot)
    comparison = compare_outright_probabilities(model_probabilities, bookmaker_snapshot)

    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    summary = {
        "captured_team_count": int(len(prepared_snapshot)),
        "captured_raw_probability_mass": float(prepared_snapshot["raw_implied_probability"].sum()),
        "top_positive_edges_vs_raw_bookmaker": comparison.head(10).loc[
            :,
            [
                "team",
                "champion_probability",
                "raw_implied_probability",
                "edge_vs_bookmaker_raw",
            ],
        ].to_dict(orient="records"),
        "top_negative_edges_vs_raw_bookmaker": comparison.sort_values(
            "edge_vs_bookmaker_raw",
            ascending=True,
            kind="stable",
        )
        .head(10)
        .loc[
            :,
            [
                "team",
                "champion_probability",
                "raw_implied_probability",
                "edge_vs_bookmaker_raw",
            ],
        ]
        .to_dict(orient="records"),
    }
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved bookmaker comparison to {output_path}")
    print(f"Saved summary to {summary_output_path}")


if __name__ == "__main__":
    main()
