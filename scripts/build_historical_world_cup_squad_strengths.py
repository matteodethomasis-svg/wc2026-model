from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.data import load_club_elo_snapshot, load_world_cup_squads_from_wikipedia
from wc2026_model.features import (
    aggregate_team_squad_strength,
    build_squad_player_club_elo_frame,
)

WORLD_CUP_SQUAD_SOURCES = {
    2014: {
        "squads_url": "https://en.wikipedia.org/wiki/2014_FIFA_World_Cup_squads",
        "club_elo_date": "2014-06-01",
    },
    2018: {
        "squads_url": "https://en.wikipedia.org/wiki/2018_FIFA_World_Cup_squads",
        "club_elo_date": "2018-06-01",
    },
    2022: {
        "squads_url": "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup_squads",
        "club_elo_date": "2022-11-15",
    },
    2026: {
        "squads_url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads",
        "club_elo_date": "2026-06-01",
    },
}


def _parse_years(value: str) -> list[int]:
    years = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not years:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated tournament year.")
    return years


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build squad-strength ratings for multiple World Cup editions by mapping "
            "final squads to Club Elo snapshots."
        )
    )
    parser.add_argument(
        "--years",
        type=_parse_years,
        default=[2014, 2018, 2022, 2026],
        help="Comma-separated World Cup years to process.",
    )
    parser.add_argument(
        "--players-output",
        default="reports/historical_world_cup_squad_players_with_club_elo.csv",
    )
    parser.add_argument(
        "--teams-output",
        default="reports/historical_world_cup_squad_strength_ratings.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/historical_world_cup_squad_strength_summary.json",
    )
    parser.add_argument("--top-player-count", type=int, default=15)
    parser.add_argument("--core-player-count", type=int, default=11)
    parser.add_argument("--star-player-count", type=int, default=3)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    selected_years = [year for year in args.years if year in WORLD_CUP_SQUAD_SOURCES]
    if not selected_years:
        raise ValueError("No valid World Cup years selected.")

    all_players: list[pd.DataFrame] = []
    all_teams: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for year in selected_years:
        config = WORLD_CUP_SQUAD_SOURCES[year]
        squads = load_world_cup_squads_from_wikipedia(config["squads_url"])
        club_elo = load_club_elo_snapshot(as_of_date=config["club_elo_date"])
        squad_players = build_squad_player_club_elo_frame(squads, club_elo)
        team_strengths = aggregate_team_squad_strength(
            squad_players,
            top_player_count=args.top_player_count,
            core_player_count=args.core_player_count,
            star_player_count=args.star_player_count,
        )

        squad_players = squad_players.assign(
            tournament_year=year,
            club_elo_snapshot_date=config["club_elo_date"],
            squads_url=config["squads_url"],
        )
        team_strengths = team_strengths.assign(
            tournament_year=year,
            club_elo_snapshot_date=config["club_elo_date"],
            squads_url=config["squads_url"],
        )

        all_players.append(squad_players)
        all_teams.append(team_strengths)
        summary_rows.append(
            {
                "tournament_year": year,
                "club_elo_snapshot_date": config["club_elo_date"],
                "team_count": int(team_strengths["team"].nunique()),
                "player_count": int(len(squad_players)),
                "mapped_player_share_overall": float(squad_players["club_elo"].notna().mean()),
                "top_5_teams": team_strengths.head(5)
                .loc[
                    :,
                    [
                        "team",
                        "squad_club_elo_rating",
                        "expected_xi_club_elo_rating",
                        "expected_xi_goalkeeper_club_elo_rating",
                        "expected_xi_attack_club_elo_rating",
                    ],
                ]
                .to_dict(orient="records"),
            }
        )

    players_output = Path(args.players_output)
    teams_output = Path(args.teams_output)
    summary_output = Path(args.summary_output)
    for path in (players_output, teams_output, summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    historical_players = pd.concat(all_players, ignore_index=True)
    historical_teams = pd.concat(all_teams, ignore_index=True)
    historical_players.to_csv(players_output, index=False)
    historical_teams.to_csv(teams_output, index=False)

    summary = {
        "years": selected_years,
        "rows": summary_rows,
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved historical squad player mapping to {players_output}")
    print(f"Saved historical squad team strengths to {teams_output}")


if __name__ == "__main__":
    main()
