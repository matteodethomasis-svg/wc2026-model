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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build World Cup 2026 squad-strength ratings by mapping final rosters "
            "to current Club Elo ratings for players' clubs."
        )
    )
    parser.add_argument(
        "--club-elo-date",
        default="2026-06-01",
        help="Snapshot date used for Club Elo ratings (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--groups-input",
        default="data/reference/wc2026_groups_actual.csv",
        help="Optional CSV used to merge group labels into the team-strength output.",
    )
    parser.add_argument(
        "--squads-output",
        default="data/interim/wc2026_squads_from_wikipedia.csv",
        help="Path to save the parsed World Cup squads table.",
    )
    parser.add_argument(
        "--club-elo-output",
        default="data/interim/club_elo_snapshot_2026-06-01.csv",
        help="Path to save the standardized Club Elo snapshot.",
    )
    parser.add_argument(
        "--players-output",
        default="reports/wc2026_squad_players_with_club_elo.csv",
        help="Path to save the player-level squad-to-club-Elo mapping.",
    )
    parser.add_argument(
        "--teams-output",
        default="reports/wc2026_squad_strength_ratings.csv",
        help="Path to save the team-level squad-strength ratings.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/wc2026_squad_strength_summary.json",
        help="Path to save a compact summary JSON.",
    )
    parser.add_argument(
        "--top-player-count",
        type=int,
        default=15,
        help="Number of top club-Elo players used in the main squad-strength rating.",
    )
    parser.add_argument(
        "--core-player-count",
        type=int,
        default=11,
        help="Number of top club-Elo players used in the core squad-strength rating.",
    )
    parser.add_argument(
        "--star-player-count",
        type=int,
        default=3,
        help="Number of top club-Elo players used in the star squad-strength rating.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    squads = load_world_cup_squads_from_wikipedia()
    club_elo = load_club_elo_snapshot(as_of_date=args.club_elo_date)
    squad_players = build_squad_player_club_elo_frame(squads, club_elo)
    team_strengths = aggregate_team_squad_strength(
        squad_players,
        top_player_count=args.top_player_count,
        core_player_count=args.core_player_count,
        star_player_count=args.star_player_count,
    )
    if Path(args.groups_input).exists():
        team_strengths = _merge_groups(team_strengths, Path(args.groups_input))

    output_paths = [
        Path(args.squads_output),
        Path(args.club_elo_output),
        Path(args.players_output),
        Path(args.teams_output),
        Path(args.summary_output),
    ]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    squads.to_csv(args.squads_output, index=False)
    club_elo.to_csv(args.club_elo_output, index=False)
    squad_players.to_csv(args.players_output, index=False)
    team_strengths.to_csv(args.teams_output, index=False)

    summary = {
        "club_elo_date": args.club_elo_date,
        "team_count": int(team_strengths["team"].nunique()),
        "player_count": int(len(squad_players)),
        "mapped_player_share_overall": float(squad_players["club_elo"].notna().mean()),
        "top_10_teams_by_squad_strength": team_strengths.head(10)
        .loc[:, ["team", "squad_club_elo_rating", "squad_club_elo_core_rating"]]
        .to_dict(orient="records"),
        "top_10_teams_by_expected_xi_strength": team_strengths.head(10)
        .loc[
            :,
            [
                "team",
                "expected_xi_club_elo_rating",
                "expected_xi_goalkeeper_club_elo_rating",
                "expected_xi_attack_club_elo_rating",
                "expected_xi_selection_score",
            ],
        ]
        .to_dict(orient="records"),
    }
    if "group" in team_strengths.columns:
        summary["group_i_snapshot"] = team_strengths.loc[
            team_strengths["group"] == "I",
            [
                "team",
                "squad_club_elo_rating",
                "expected_xi_club_elo_rating",
                "expected_xi_formation",
                "expected_xi_goalkeeper_club_elo_rating",
                "expected_xi_defense_club_elo_rating",
                "expected_xi_midfield_club_elo_rating",
                "expected_xi_attack_club_elo_rating",
                "mapped_player_share",
                "expected_xi_mapped_player_share",
            ],
        ].sort_values("expected_xi_club_elo_rating", ascending=False, kind="stable").to_dict(orient="records")

    Path(args.summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved squads to {args.squads_output}")
    print(f"Saved Club Elo snapshot to {args.club_elo_output}")
    print(f"Saved player squad-strength mapping to {args.players_output}")
    print(f"Saved team squad-strength ratings to {args.teams_output}")


def _merge_groups(team_strengths: pd.DataFrame, groups_path: Path) -> pd.DataFrame:
    groups = pd.read_csv(groups_path)
    groups = groups.loc[:, ["group", "team"]].copy()
    groups["group"] = groups["group"].astype(str)
    groups["team"] = groups["team"].astype(str)
    return team_strengths.merge(groups, on="team", how="left")


if __name__ == "__main__":
    main()
