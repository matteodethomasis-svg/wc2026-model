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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose a World Cup 2026 group by comparing direct match probabilities with Monte Carlo group results."
    )
    parser.add_argument("--group", required=True, help="Group letter, e.g. I")
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
        default="reports/group_diagnosis.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    with Path(args.model_input).open("rb") as file_handle:
        model = pickle.load(file_handle)

    groups = _load_groups(Path(args.groups_input), model=model)
    elo_ratings = _load_elo_ratings(Path(args.elo_ratings_input), model=model)
    group_name = str(args.group).strip().upper()
    schedule = build_group_stage_schedule(groups)
    group_schedule = schedule.loc[schedule["group"] == group_name].reset_index(drop=True)
    if group_schedule.empty:
        raise ValueError(f"Group {group_name} not found in groups input.")

    match_rows: list[dict[str, object]] = []
    expected_points = {team: 0.0 for team in groups[group_name]}
    for row in group_schedule.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        elo_diff = float(elo_ratings.get(home_team, 1500.0) - elo_ratings.get(away_team, 1500.0))
        probabilities = model.predict_outcome_probabilities(
            home_team,
            away_team,
            neutral_site=True,
            elo_diff_pre=elo_diff,
            max_goals=args.max_goals,
        )
        match_rows.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "home_win_probability": probabilities.home,
                "draw_probability": probabilities.draw,
                "away_win_probability": probabilities.away,
            }
        )
        expected_points[home_team] += (3.0 * probabilities.home) + probabilities.draw
        expected_points[away_team] += (3.0 * probabilities.away) + probabilities.draw

    monte_carlo = _simulate_group_points(
        model=model,
        group_schedule=group_schedule,
        elo_ratings=elo_ratings,
        simulations=args.simulations,
        random_state=args.random_state,
        max_goals=args.max_goals,
    )

    output = {
        "group": group_name,
        "simulations": args.simulations,
        "matches": match_rows,
        "expected_points_from_match_probabilities": expected_points,
        "monte_carlo_average_points": monte_carlo["average_points"].to_dict(),
        "monte_carlo_group_winner_probability": monte_carlo["group_winner_probability"].to_dict(),
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


def _simulate_group_points(
    *,
    model: object,
    group_schedule: pd.DataFrame,
    elo_ratings: dict[str, float],
    simulations: int,
    random_state: int,
    max_goals: int,
) -> pd.DataFrame:
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

    rows = []
    for team in teams:
        rows.append(
            {
                "team": team,
                "average_points": group_points_total[team] / simulations,
                "group_winner_probability": group_winner_total[team] / simulations,
            }
        )
    return pd.DataFrame(rows).set_index("team")


if __name__ == "__main__":
    main()
