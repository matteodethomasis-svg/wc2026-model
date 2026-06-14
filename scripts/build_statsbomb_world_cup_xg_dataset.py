from __future__ import annotations

import argparse
import json
from pathlib import Path

from wc2026_model.data import (
    build_statsbomb_competition_match_features,
    build_statsbomb_team_xg_summary,
)

DEFAULT_COMPETITION_SET = {
    "world_cup_modern": ("FIFA World Cup",),
    "men_major_tournaments": (
        "FIFA World Cup",
        "UEFA Euro",
        "Copa America",
        "African Cup of Nations",
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a StatsBomb open-data international match feature dataset with xG and event stats."
    )
    parser.add_argument(
        "--competition-names",
        default="FIFA World Cup",
        help="Comma-separated competition names to filter from StatsBomb competitions.",
    )
    parser.add_argument(
        "--competition-set",
        choices=sorted(DEFAULT_COMPETITION_SET),
        default=None,
        help="Optional preset competition bundle.",
    )
    parser.add_argument(
        "--competition-gender",
        default="male",
        help="Optional competition gender filter. Use an empty string to disable.",
    )
    parser.add_argument(
        "--season-names",
        default="",
        help="Comma-separated season names to include. Empty means all available seasons.",
    )
    parser.add_argument(
        "--output",
        default="data/interim/statsbomb_international_match_features.csv",
        help="Path to save the match-level feature dataset.",
    )
    parser.add_argument(
        "--team-summary-output",
        default="reports/statsbomb_international_team_xg_summary.csv",
        help="Path to save the team-level xG summary.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/statsbomb_international_xg_summary.json",
        help="Path to save a JSON summary of the extract.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    output_path = Path(args.output)
    team_summary_output = Path(args.team_summary_output)
    summary_output = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    team_summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    competition_gender = args.competition_gender.strip() or None
    competition_names = _resolve_competition_names(
        competition_set=args.competition_set,
        raw_value=args.competition_names,
    )
    season_names = _parse_season_names(args.season_names)

    matches = build_statsbomb_competition_match_features(
        competition_names=competition_names,
        competition_gender=competition_gender,
        season_names=season_names,
    )
    matches.to_csv(output_path, index=False)

    team_summary = build_statsbomb_team_xg_summary(matches)
    team_summary.to_csv(team_summary_output, index=False)

    summary = {
        "competition_names": list(competition_names),
        "competition_set": args.competition_set,
        "competition_gender": competition_gender,
        "competitions_in_output": sorted(matches["tournament"].dropna().astype(str).unique().tolist()),
        "season_names": sorted(matches["source_season_name"].dropna().astype(str).unique().tolist()),
        "match_count": int(len(matches)),
        "team_count": int(len(set(matches["home_team"]).union(set(matches["away_team"])))),
        "date_min": matches["match_date"].min().strftime("%Y-%m-%d") if not matches.empty else None,
        "date_max": matches["match_date"].max().strftime("%Y-%m-%d") if not matches.empty else None,
        "mean_home_xg": float(matches["home_xg"].mean()) if not matches.empty else None,
        "mean_away_xg": float(matches["away_xg"].mean()) if not matches.empty else None,
        "output": str(output_path),
        "team_summary_output": str(team_summary_output),
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved match features to {output_path}")
    print(f"Saved team summary to {team_summary_output}")
    print(f"Saved extract summary to {summary_output}")


def _parse_season_names(raw_value: str) -> tuple[str, ...] | None:
    season_names = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    return season_names or None


def _resolve_competition_names(
    *,
    competition_set: str | None,
    raw_value: str,
) -> tuple[str, ...]:
    if competition_set:
        return DEFAULT_COMPETITION_SET[competition_set]
    competition_names = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    if not competition_names:
        raise ValueError("At least one competition name must be provided.")
    return competition_names


if __name__ == "__main__":
    main()
