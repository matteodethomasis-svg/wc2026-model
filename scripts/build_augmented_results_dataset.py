from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from wc2026_model.data import (
    FOOTBALL_DATA_DEFAULT_COMPETITION_CODES,
    combine_standardized_results,
    download_cup26_open_results_json,
    fetch_world_cup_results,
    download_international_results_csv,
    fetch_football_data_matches,
    get_football_data_api_token,
    load_cup26_open_results,
    load_international_results,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an augmented international-results dataset with recent 2025-2026 coverage."
    )
    parser.add_argument(
        "--historical-input",
        default="data/raw/international_results.csv",
        help="Path to the historical CSV from the martj42/international_results repo.",
    )
    parser.add_argument(
        "--auto-download-historical",
        action="store_true",
        help="Download the historical CSV automatically if it does not exist.",
    )
    parser.add_argument(
        "--skip-cup26-open",
        action="store_true",
        help="Skip the open recent-results supplement.",
    )
    parser.add_argument(
        "--allow-unknown-teams-in-cup26-open",
        action="store_true",
        help="Keep cup26-open rows even if one team is outside the historical national-team universe.",
    )
    parser.add_argument(
        "--cup26-raw-output",
        default="data/raw/cup26_open_results.json",
        help="Path to cache the raw open recent-results JSON.",
    )
    parser.add_argument(
        "--skip-espn",
        action="store_true",
        help="Skip the free ESPN live WC2026 results feed.",
    )
    parser.add_argument(
        "--espn-start-date",
        default="2026-06-10",
        help="First date (inclusive) to pull WC2026 results from ESPN.",
    )
    parser.add_argument(
        "--espn-end-date",
        default=date.today().isoformat(),
        help="Last date (inclusive) to pull WC2026 results from ESPN (default: today).",
    )
    parser.add_argument(
        "--include-football-data",
        action="store_true",
        help="Append matches from football-data.org using FOOTBALL_DATA_API_TOKEN.",
    )
    parser.add_argument(
        "--football-data-competition-codes",
        default=",".join(FOOTBALL_DATA_DEFAULT_COMPETITION_CODES),
        help="Comma-separated football-data.org competition codes.",
    )
    parser.add_argument(
        "--football-data-season",
        type=int,
        default=None,
        help="Optional season filter for football-data.org.",
    )
    parser.add_argument(
        "--football-data-date-from",
        default="2025-01-01",
        help="Optional dateFrom filter for football-data.org.",
    )
    parser.add_argument(
        "--football-data-date-to",
        default=None,
        help="Optional dateTo filter for football-data.org.",
    )
    parser.add_argument(
        "--football-data-output",
        default="data/interim/football_data_recent_matches.csv",
        help="Path to save the standardized football-data.org extract if used.",
    )
    parser.add_argument(
        "--output",
        default="data/interim/international_results_augmented.csv",
        help="Path to save the merged standardized dataset.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/augmented_results_summary.json",
        help="Path to save a summary JSON.",
    )
    return parser


def _ensure_historical_input(path: Path, auto_download: bool) -> Path:
    if path.exists():
        return path
    if not auto_download:
        raise FileNotFoundError(
            f"Historical input CSV not found at {path}. Re-run with --auto-download-historical."
        )
    return download_international_results_csv(path)


def main() -> None:
    args = _build_parser().parse_args()

    historical_input = _ensure_historical_input(
        Path(args.historical_input), args.auto_download_historical
    )
    output_path = Path(args.output)
    summary_output = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    historical = load_international_results(historical_input).copy()
    historical["source"] = "historical_csv"
    historical_team_universe = set(historical["home_team"]).union(set(historical["away_team"]))

    frames: list[pd.DataFrame] = [historical]
    component_summaries: dict[str, dict[str, object]] = {
        "historical_csv": _frame_summary(historical),
    }

    if not args.skip_cup26_open:
        cup26_raw_output = Path(args.cup26_raw_output)
        download_cup26_open_results_json(cup26_raw_output)
        cup26_open = load_cup26_open_results(cup26_raw_output)
        removed_unknown_team_rows = 0
        if not args.allow_unknown_teams_in_cup26_open:
            cup26_open, removed_unknown_team_rows = _filter_to_known_team_universe(
                cup26_open,
                historical_team_universe,
            )
        frames.append(cup26_open)
        component_summaries["cup26_open"] = _frame_summary(cup26_open) | {
            "removed_unknown_team_rows": removed_unknown_team_rows
        }

    if not args.skip_espn:
        try:
            espn = fetch_world_cup_results(args.espn_start_date, args.espn_end_date)
            if not espn.empty:
                espn, removed_espn_unknown = _filter_to_known_team_universe(
                    espn, historical_team_universe
                )
                frames.append(espn)
                component_summaries["espn"] = _frame_summary(espn) | {
                    "removed_unknown_team_rows": removed_espn_unknown
                }
        except Exception as error:  # never let a live-feed hiccup break the build
            component_summaries["espn"] = {"error": str(error)}

    if args.include_football_data:
        football_data = fetch_football_data_matches(
            api_token=get_football_data_api_token(),
            competition_codes=_parse_competition_codes(args.football_data_competition_codes),
            season=args.football_data_season,
            date_from=args.football_data_date_from,
            date_to=args.football_data_date_to,
        )
        football_data_output = Path(args.football_data_output)
        football_data_output.parent.mkdir(parents=True, exist_ok=True)
        football_data.to_csv(football_data_output, index=False)
        frames.append(football_data)
        component_summaries["football_data_api"] = _frame_summary(football_data)

    # ESPN is the freshest, most reliable tournament feed -> highest dedupe priority.
    merged = combine_standardized_results(
        frames,
        source_priority=("espn", "football_data_api", "cup26_open", "historical_csv"),
    )
    merged.to_csv(output_path, index=False)

    summary = {
        "components": component_summaries,
        "merged": _frame_summary(merged),
        "output": str(output_path),
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved merged dataset to {output_path}")
    print(f"Saved summary to {summary_output}")


def _parse_competition_codes(raw_codes: str) -> tuple[str, ...]:
    return tuple(code.strip() for code in raw_codes.split(",") if code.strip())


def _frame_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "match_count": 0,
            "date_min": None,
            "date_max": None,
            "team_count": 0,
            "competition_count": 0,
        }
    return {
        "match_count": int(len(frame)),
        "date_min": frame["match_date"].min().strftime("%Y-%m-%d"),
        "date_max": frame["match_date"].max().strftime("%Y-%m-%d"),
        "team_count": int(len(set(frame["home_team"]).union(set(frame["away_team"])))),
        "competition_count": int(frame["tournament"].nunique()),
    }


def _filter_to_known_team_universe(
    frame: pd.DataFrame,
    known_teams: set[str],
) -> tuple[pd.DataFrame, int]:
    mask = frame["home_team"].isin(known_teams) & frame["away_team"].isin(known_teams)
    filtered = frame.loc[mask].copy()
    removed = int((~mask).sum())
    return filtered.reset_index(drop=True), removed


if __name__ == "__main__":
    main()
