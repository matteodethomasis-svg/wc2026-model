from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from wc2026_model.data import canonicalize_team_name
from wc2026_model.tournament import build_group_stage_schedule, rank_group_standings
from wc2026_model.tournament.simulation import sample_scoreline_from_matrix


def _parse_bump_grid(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated bump.")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure how a team-specific Elo bump changes World Cup group-win probability."
    )
    parser.add_argument("--group", required=True, help="Group letter, e.g. I")
    parser.add_argument("--team", required=True, help="Target team name to adjust.")
    parser.add_argument(
        "--elo-bumps",
        type=_parse_bump_grid,
        default=[0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0],
        help="Comma-separated Elo bump grid, e.g. 0,25,50,75,100.",
    )
    parser.add_argument(
        "--groups-input",
        default="data/reference/wc2026_groups_actual.csv",
    )
    parser.add_argument(
        "--model-input",
        default="models/baseline_dixon_coles_elo.pkl",
    )
    parser.add_argument(
        "--elo-ratings-input",
        default="reports/baseline_latest_elo_ratings.csv",
    )
    parser.add_argument("--simulations", type=int, default=20000)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument(
        "--output",
        default="reports/group_team_elo_sensitivity.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    with Path(args.model_input).open("rb") as file_handle:
        model = pickle.load(file_handle)

    groups = _load_groups(Path(args.groups_input), model=model)
    elo_ratings = _load_elo_ratings(Path(args.elo_ratings_input), model=model)

    group_name = str(args.group).strip().upper()
    target_team = canonicalize_team_name(str(args.team).strip())
    if group_name not in groups:
        raise ValueError(f"Group {group_name} not found in groups input.")
    if target_team not in groups[group_name]:
        raise ValueError(f"Team {target_team} not found in group {group_name}.")

    schedule = build_group_stage_schedule(groups)
    group_schedule = schedule.loc[schedule["group"] == group_name].reset_index(drop=True)
    baseline_elo = float(elo_ratings.get(target_team, 1500.0))

    rows: list[dict[str, object]] = []
    for elo_bump in args.elo_bumps:
        adjusted_elo_ratings = dict(elo_ratings)
        adjusted_elo_ratings[target_team] = baseline_elo + float(elo_bump)
        match_probabilities, expected_points = _compute_group_probabilities(
            model=model,
            group_schedule=group_schedule,
            elo_ratings=adjusted_elo_ratings,
            max_goals=args.max_goals,
        )
        monte_carlo = _simulate_group_points(
            model=model,
            group_schedule=group_schedule,
            elo_ratings=adjusted_elo_ratings,
            simulations=args.simulations,
            random_state=args.random_state,
            max_goals=args.max_goals,
        )
        target_matches = [
            match
            for match in match_probabilities
            if match["home_team"] == target_team or match["away_team"] == target_team
        ]
        rows.append(
            {
                "elo_bump": float(elo_bump),
                "target_elo_rating": float(adjusted_elo_ratings[target_team]),
                "target_expected_points": float(expected_points[target_team]),
                "target_group_winner_probability": float(
                    monte_carlo["group_winner_probability"][target_team]
                ),
                "target_matches": target_matches,
            }
        )

    output = {
        "group": group_name,
        "team": target_team,
        "baseline_elo_rating": baseline_elo,
        "simulations": args.simulations,
        "rows": rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def _load_groups(path: Path, *, model: object) -> dict[str, list[str]]:
    groups_frame = pd.read_csv(path)
    groups_frame["group"] = groups_frame["group"].astype(str)
    groups_frame["team"] = groups_frame["team"].astype(str).map(canonicalize_team_name)
    if "slot" in groups_frame.columns:
        groups_frame = groups_frame.sort_values(["group", "slot"], kind="stable")

    model_team_lookup = {
        canonicalize_team_name(str(team)): str(team)
        for team in getattr(model, "teams", [])
    }
    groups_frame["team"] = groups_frame["team"].map(lambda team: model_team_lookup.get(team, team))
    return groups_frame.groupby("group", sort=True)["team"].apply(list).to_dict()


def _load_elo_ratings(path: Path, *, model: object) -> dict[str, float]:
    ratings_frame = pd.read_csv(path)
    model_team_lookup = {
        canonicalize_team_name(str(team)): str(team)
        for team in getattr(model, "teams", [])
    }
    ratings_frame["team"] = ratings_frame["team"].astype(str).map(canonicalize_team_name)
    ratings_frame["team"] = ratings_frame["team"].map(lambda team: model_team_lookup.get(team, team))
    return {
        str(row.team): float(row.elo_rating)
        for row in ratings_frame.loc[:, ["team", "elo_rating"]].itertuples(index=False)
    }


def _compute_group_probabilities(
    *,
    model: object,
    group_schedule: pd.DataFrame,
    elo_ratings: dict[str, float],
    max_goals: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    teams = sorted(set(group_schedule["home_team"]).union(group_schedule["away_team"]))
    expected_points = {team: 0.0 for team in teams}
    match_rows: list[dict[str, object]] = []

    for row in group_schedule.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        elo_diff = float(elo_ratings.get(home_team, 1500.0) - elo_ratings.get(away_team, 1500.0))
        probabilities = model.predict_outcome_probabilities(
            home_team,
            away_team,
            neutral_site=True,
            elo_diff_pre=elo_diff,
            max_goals=max_goals,
        )
        match_rows.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "home_win_probability": float(probabilities.home),
                "draw_probability": float(probabilities.draw),
                "away_win_probability": float(probabilities.away),
            }
        )
        expected_points[home_team] += (3.0 * float(probabilities.home)) + float(probabilities.draw)
        expected_points[away_team] += (3.0 * float(probabilities.away)) + float(probabilities.draw)

    return match_rows, expected_points


def _simulate_group_points(
    *,
    model: object,
    group_schedule: pd.DataFrame,
    elo_ratings: dict[str, float],
    simulations: int,
    random_state: int,
    max_goals: int,
) -> dict[str, dict[str, float]]:
    teams = sorted(set(group_schedule["home_team"]).union(group_schedule["away_team"]))
    group_points_total = {team: 0.0 for team in teams}
    group_winner_total = {team: 0 for team in teams}
    rng = np.random.default_rng(random_state)

    for _ in range(simulations):
        simulated_matches = []
        for row in group_schedule.itertuples(index=False):
            home_team = str(row.home_team)
            away_team = str(row.away_team)
            elo_diff = float(elo_ratings.get(home_team, 1500.0) - elo_ratings.get(away_team, 1500.0))
            score_matrix = model.predict_score_matrix(
                home_team,
                away_team,
                neutral_site=True,
                elo_diff_pre=elo_diff,
                max_goals=max_goals,
            )
            home_goals, away_goals = sample_scoreline_from_matrix(score_matrix, rng)
            simulated_matches.append(
                {
                    "group": str(row.group),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                }
            )

        standings = rank_group_standings(pd.DataFrame.from_records(simulated_matches))
        for standing_row in standings.itertuples(index=False):
            team = str(standing_row.team)
            group_points_total[team] += float(standing_row.points)
            if int(standing_row.group_rank) == 1:
                group_winner_total[team] += 1

    average_points = {
        team: group_points_total[team] / simulations
        for team in teams
    }
    group_winner_probability = {
        team: group_winner_total[team] / simulations
        for team in teams
    }
    return {
        "average_points": average_points,
        "group_winner_probability": group_winner_probability,
    }


if __name__ == "__main__":
    main()
