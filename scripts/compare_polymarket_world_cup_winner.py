from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.markets.polymarket import compare_world_cup_winner_probabilities


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare our World Cup outright probabilities against the active Polymarket winner market."
    )
    parser.add_argument(
        "--model-probabilities-input",
        default="reports/wc2026_simulation_probabilities.csv",
    )
    parser.add_argument(
        "--market-probabilities-input",
        default="data/interim/polymarket_world_cup_winner.csv",
    )
    parser.add_argument(
        "--output",
        default="reports/polymarket_world_cup_winner_comparison.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/polymarket_world_cup_winner_comparison_summary.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    model_probabilities = pd.read_csv(args.model_probabilities_input)
    market_probabilities = pd.read_csv(args.market_probabilities_input)
    comparison = compare_world_cup_winner_probabilities(model_probabilities, market_probabilities)

    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    summary = {
        "team_count": int(len(comparison)),
        "top_positive_edges": comparison.head(10).loc[
            :, ["team", "champion_probability", "market_probability", "edge_vs_market"]
        ].to_dict(orient="records"),
        "top_negative_edges": comparison.sort_values(
            "edge_vs_market",
            ascending=True,
            kind="stable",
        )
        .head(10)
        .loc[:, ["team", "champion_probability", "market_probability", "edge_vs_market"]]
        .to_dict(orient="records"),
    }
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved comparison report to {output_path}")
    print(f"Saved summary to {summary_output_path}")


if __name__ == "__main__":
    main()
