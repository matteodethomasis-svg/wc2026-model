from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.data import (
    aggregate_team_availability_features,
    build_live_squad_intelligence,
    build_live_squad_intelligence_summary,
    load_expected_lineups_feed,
    load_flat_file_table,
    load_injuries_feed,
    load_world_cup_squads_from_wikipedia,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a normalized World Cup 2026 live squad-intelligence layer "
            "from official squads, expected lineups, and injury reports."
        )
    )
    parser.add_argument(
        "--official-squads-input",
        default="data/interim/wc2026_squads_from_wikipedia.csv",
        help=(
            "Flat CSV/JSON file with official squad rows. "
            "If missing, the script falls back to the Wikipedia squad parser."
        ),
    )
    parser.add_argument(
        "--expected-lineups-input",
        default=None,
        help=(
            "Optional flat CSV/JSON export for predicted starting lineups. "
            "Minimal columns: team, player. Recommended extras: fixture_id, match_date, "
            "position, formation, is_expected_starter, lineup_confidence."
        ),
    )
    parser.add_argument(
        "--expected-lineups-provider",
        choices=("auto", "flat", "sportmonks"),
        default="auto",
        help=(
            "How to interpret the expected-lineups input. "
            "'auto' will detect Sportmonks nested JSON and otherwise fall back to flat CSV/JSON."
        ),
    )
    parser.add_argument(
        "--injuries-input",
        default=None,
        help=(
            "Optional flat CSV/JSON export for injury / availability reports. "
            "Minimal columns: team, player. Recommended extras: status, reason, "
            "report_date, expected_return_date, availability_status."
        ),
    )
    parser.add_argument(
        "--injuries-provider",
        choices=("auto", "flat", "api_football"),
        default="auto",
        help=(
            "How to interpret the injuries input. "
            "'auto' will detect API-Football nested JSON and otherwise fall back to flat CSV/JSON."
        ),
    )
    parser.add_argument(
        "--official-source-label",
        default="official_wikipedia_squads",
    )
    parser.add_argument(
        "--expected-lineup-source-label",
        default="sportmonks_expected_lineups",
    )
    parser.add_argument(
        "--injury-source-label",
        default="api_football_injuries",
    )
    parser.add_argument(
        "--player-output",
        default="reports/wc2026_live_squad_intelligence.csv",
    )
    parser.add_argument(
        "--team-output",
        default="reports/wc2026_team_availability_features.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/wc2026_live_squad_intelligence_summary.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    official_squads = _load_official_squads(Path(args.official_squads_input))
    expected_lineups = (
        load_expected_lineups_feed(
            args.expected_lineups_input,
            provider=args.expected_lineups_provider,
        )
        if args.expected_lineups_input
        else None
    )
    injuries = (
        load_injuries_feed(
            args.injuries_input,
            provider=args.injuries_provider,
        )
        if args.injuries_input
        else None
    )

    player_intelligence = build_live_squad_intelligence(
        official_squads,
        expected_lineups=expected_lineups,
        injuries=injuries,
        official_source_label=args.official_source_label,
        expected_lineup_source_label=args.expected_lineup_source_label,
        injury_source_label=args.injury_source_label,
    )
    team_features = aggregate_team_availability_features(player_intelligence)
    summary = build_live_squad_intelligence_summary(
        player_intelligence,
        expected_lineups=expected_lineups,
        injuries=injuries,
    )
    summary["official_squad_input_rows"] = int(len(official_squads))
    summary["team_feature_rows"] = int(len(team_features))
    if not team_features.empty:
        sort_columns = [
            "unavailable_expected_starter_count",
            "doubtful_expected_starter_count",
            "expected_starter_count",
        ]
        summary["top_team_availability_risks"] = (
            team_features.sort_values(
                sort_columns,
                ascending=[False, False, False],
                kind="stable",
            )
            .head(10)
            .to_dict(orient="records")
        )
    else:
        summary["top_team_availability_risks"] = []

    player_output = Path(args.player_output)
    team_output = Path(args.team_output)
    summary_output = Path(args.summary_output)
    for path in (player_output, team_output, summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    player_intelligence.to_csv(player_output, index=False)
    team_features.to_csv(team_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved player-level live squad intelligence to {player_output}")
    print(f"Saved team-level availability features to {team_output}")


def _load_official_squads(path: Path) -> pd.DataFrame:
    if path.exists():
        return load_flat_file_table(path)
    return load_world_cup_squads_from_wikipedia()


if __name__ == "__main__":
    main()
