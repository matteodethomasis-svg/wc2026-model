from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.data import (
    get_api_football_api_key,
    get_api_football_api_key_header,
    get_api_football_host,
    read_provider_team_ids,
    save_api_football_injuries_by_team_ids_outputs,
    save_api_football_injuries_outputs,
    select_provider_team_ids_for_fixture_window,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download API-Football injuries and save raw JSON plus normalized CSV."
    )
    parser.add_argument(
        "--league",
        type=int,
        default=None,
        help="Optional API-Football league ID.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Optional season year.",
    )
    parser.add_argument(
        "--fixture",
        type=int,
        default=None,
        help="Optional fixture ID.",
    )
    parser.add_argument(
        "--team",
        type=int,
        default=None,
        help="Optional team ID.",
    )
    parser.add_argument(
        "--team-ids",
        default=None,
        help="Optional comma-separated API-Football team IDs.",
    )
    parser.add_argument(
        "--registry-input",
        default=None,
        help=(
            "Optional provider registry CSV. When supplied, API-Football IDs are read from its "
            "'api_football_team_id' column."
        ),
    )
    parser.add_argument(
        "--player",
        type=int,
        default=None,
        help="Optional player ID.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Optional date filter in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--timezone",
        default=None,
        help="Optional timezone string accepted by API-Football.",
    )
    parser.add_argument(
        "--fixtures-input",
        default="data/raw/international_results.csv",
        help="Raw results/fixtures CSV used to target only upcoming World Cup teams.",
    )
    parser.add_argument(
        "--tournament",
        default="FIFA World Cup",
        help="Tournament filter used with --upcoming-window-days.",
    )
    parser.add_argument(
        "--start-date",
        default="2026-06-12",
        help="Start date used with --upcoming-window-days.",
    )
    parser.add_argument(
        "--upcoming-window-days",
        type=int,
        default=None,
        help="Optional rolling window of upcoming matchdays used to limit team requests.",
    )
    parser.add_argument(
        "--max-teams",
        type=int,
        default=None,
        help="Optional cap on the number of targeted teams after fixture-window filtering.",
    )
    parser.add_argument(
        "--free-plan",
        action="store_true",
        help="Convenience mode for API-Football Free: target only upcoming teams over a short window.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="Optional page number. Omit to auto-paginate through all pages.",
    )
    parser.add_argument(
        "--raw-output",
        default="data/interim/api_football_injuries_raw.json",
        help="Where to save the raw JSON payload.",
    )
    parser.add_argument(
        "--csv-output",
        default="data/interim/api_football_injuries.csv",
        help="Where to save the normalized CSV.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/api_football_injuries_download_summary.json",
        help="Where to save a compact JSON summary.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.free_plan:
        if args.upcoming_window_days is None:
            args.upcoming_window_days = 2
        if args.max_teams is None:
            args.max_teams = 16

    api_key = get_api_football_api_key()
    api_key_header = get_api_football_api_key_header()
    api_host = get_api_football_host()
    registry_path = Path(args.registry_input) if args.registry_input else None
    team_ids = _resolve_team_ids(
        csv_team_ids=args.team_ids,
        registry_input=registry_path,
    )
    targeting = _apply_fixture_window_targeting(
        team_ids=team_ids,
        registry_input=registry_path,
        fixtures_input=Path(args.fixtures_input),
        tournament=args.tournament,
        start_date=args.start_date,
        upcoming_window_days=args.upcoming_window_days,
        max_teams=args.max_teams,
    )
    team_ids = targeting["team_ids"]
    targeting_summary = targeting["targeting_summary"]
    if (
        targeting_summary is not None
        and not team_ids
        and args.team is None
        and args.fixture is None
        and args.player is None
    ):
        missing_teams = targeting_summary.get("missing_teams") or []
        raise ValueError(
            "No API-Football team IDs were available for the targeted fixture window. "
            "Populate the registry for these teams first: "
            + ", ".join(map(str, missing_teams))
        )

    raw_output = Path(args.raw_output)
    csv_output = Path(args.csv_output)
    summary_output = Path(args.summary_output)
    for path in (raw_output, csv_output, summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    if team_ids:
        save_api_football_injuries_by_team_ids_outputs(
            raw_destination=raw_output,
            csv_destination=csv_output,
            team_ids=team_ids,
            api_key=api_key,
            league=args.league,
            season=args.season,
            fixture=args.fixture,
            date=args.date,
            timezone=args.timezone,
            page=args.page,
            api_key_header=api_key_header,
            api_host=api_host,
        )
    else:
        save_api_football_injuries_outputs(
            raw_destination=raw_output,
            csv_destination=csv_output,
            api_key=api_key,
            league=args.league,
            season=args.season,
            fixture=args.fixture,
            team=args.team,
            player=args.player,
            date=args.date,
            timezone=args.timezone,
            page=args.page,
            api_key_header=api_key_header,
            api_host=api_host,
        )

    dataframe = pd.read_csv(csv_output)
    summary = {
        "row_count": int(len(dataframe)),
        "team_count": int(dataframe["team"].nunique()) if "team" in dataframe.columns else 0,
        "fixture_count": int(dataframe["fixture_id"].nunique()) if "fixture_id" in dataframe.columns else 0,
        "unavailable_rows": int(
            dataframe["availability_status"].astype(str).eq("unavailable").sum()
            if "availability_status" in dataframe.columns
            else 0
        ),
        "doubtful_rows": int(
            dataframe["availability_status"].astype(str).eq("doubtful").sum()
            if "availability_status" in dataframe.columns
            else 0
        ),
        "query": {
            "league": args.league,
            "season": args.season,
            "fixture": args.fixture,
            "team": args.team,
            "team_ids": team_ids,
            "player": args.player,
            "date": args.date,
            "timezone": args.timezone,
            "page": args.page,
        },
        "targeting": targeting_summary,
        "raw_output": str(raw_output),
        "csv_output": str(csv_output),
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved raw API-Football injuries to {raw_output}")
    print(f"Saved normalized API-Football injuries to {csv_output}")


def _resolve_team_ids(
    *,
    csv_team_ids: str | None,
    registry_input: Path | None,
) -> list[int]:
    resolved: list[int] = []
    if csv_team_ids:
        resolved.extend(
            int(team_id.strip())
            for team_id in str(csv_team_ids).split(",")
            if team_id.strip() != ""
        )
    if registry_input is not None:
        resolved.extend(read_provider_team_ids(registry_input, provider="api_football"))
    return list(dict.fromkeys(resolved))


def _apply_fixture_window_targeting(
    *,
    team_ids: list[int],
    registry_input: Path | None,
    fixtures_input: Path,
    tournament: str,
    start_date: str,
    upcoming_window_days: int | None,
    max_teams: int | None,
) -> dict[str, object]:
    if upcoming_window_days is None:
        return {
            "team_ids": team_ids,
            "targeting_summary": None,
        }
    if registry_input is None:
        raise ValueError("--upcoming-window-days requires --registry-input.")

    registry = pd.read_csv(registry_input)
    targeted = select_provider_team_ids_for_fixture_window(
        registry,
        provider="api_football",
        fixtures_input=fixtures_input,
        tournament=tournament,
        start_date=start_date,
        window_days=upcoming_window_days,
        max_teams=max_teams,
    )
    targeted_ids = [int(team_id) for team_id in targeted["team_ids"]]
    merged_ids = list(dict.fromkeys(targeted_ids + team_ids))
    return {
        "team_ids": merged_ids,
        "targeting_summary": {
            key: value
            for key, value in targeted.items()
            if key != "window_frame"
        },
    }


if __name__ == "__main__":
    main()
