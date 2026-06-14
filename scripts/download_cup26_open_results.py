from __future__ import annotations

import argparse
import json
from pathlib import Path

from wc2026_model.data import (
    download_cup26_open_results_json,
    load_cup26_open_results,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the open recent international-results dataset used by a public WC2026 model."
    )
    parser.add_argument(
        "--raw-output",
        default="data/raw/cup26_open_results.json",
        help="Path to save the raw JSON payload.",
    )
    parser.add_argument(
        "--standardized-output",
        default="data/interim/cup26_open_results_standardized.csv",
        help="Path to save the standardized CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the raw JSON if it already exists.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    raw_output = Path(args.raw_output)
    standardized_output = Path(args.standardized_output)

    download_cup26_open_results_json(raw_output, overwrite=args.overwrite)
    standardized = load_cup26_open_results(raw_output)

    standardized_output.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(standardized_output, index=False)

    summary = {
        "match_count": int(len(standardized)),
        "date_min": standardized["match_date"].min().strftime("%Y-%m-%d"),
        "date_max": standardized["match_date"].max().strftime("%Y-%m-%d"),
        "team_count": int(
            len(set(standardized["home_team"]).union(set(standardized["away_team"])))
        ),
        "competition_count": int(standardized["tournament"].nunique()),
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved raw JSON to {raw_output}")
    print(f"Saved standardized CSV to {standardized_output}")


if __name__ == "__main__":
    main()
