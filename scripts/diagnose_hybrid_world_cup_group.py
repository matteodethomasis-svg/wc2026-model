from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from wc2026_model.data import canonicalize_team_name
from wc2026_model.evaluation import EloMultinomialBenchmark
from wc2026_model.models import (
    blend_three_way_probabilities,
    reweight_score_matrix_to_outcomes,
)
from wc2026_model.tournament import build_group_stage_schedule, rank_group_standings
from wc2026_model.tournament.simulation import sample_scoreline_from_matrix


def _parse_weight_grid(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated weight.")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose a World Cup group under a Dixon-Coles plus Elo outcome blend."
    )
    parser.add_argument("--group", required=True, help="Group letter, e.g. I")
    parser.add_argument(
        "--alpha-grid",
        type=_parse_weight_grid,
        default=[1.0, 0.9, 0.8, 0.75, 0.7],
        help="Comma-separated weights on Dixon-Coles probabilities. Elo gets 1-alpha.",
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
    parser.add_argument(
        "--training-frame-input",
        default="reports/baseline_training_frame.csv",
    )
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument(
        "--output",
        default="reports/group_hybrid_diagnosis.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    with Path(args.model_input).open("rb") as file_handle:
        model = pickle.load(file_handle)

    groups = _load_groups(Path(args.groups_input), model=model)
    elo_ratings = _load_elo_ratings(Path(args.elo_ratings_input), model=model)
    training_frame = pd.read_csv(args.training_frame_input)
    elo_benchmark = EloMultinomialBenchmark.fit(training_frame)

    group_name = str(args.group).strip().upper()
    schedule = build_group_stage_schedule(groups)
    group_schedule = schedule.loc[schedule["group"] == group_name].reset_index(drop=True)
    if group_schedule.empty:
        raise ValueError(f"Group {group_name} not found in groups input.")

    rows: list[dict[str, object]] = []
    for alpha_on_dixon_coles in args.alpha_grid:
        match_rows, expected_points = _compute_group_probabilities(
            model=model,
            elo_benchmark=elo_benchmark,
            group_schedule=group_schedule,
            elo_ratings=elo_ratings,
            alpha_on_dixon_coles=float(alpha_on_dixon_coles),
            max_goals=args.max_goals,
        )
        monte_carlo = _simulate_group_points(
            model=model,
            elo_benchmark=elo_benchmark,
            group_schedule=group_schedule,
            elo_ratings=elo_ratings,
            alpha_on_dixon_coles=float(alpha_on_dixon_coles),
            simulations=args.simulations,
            random_state=args.random_state,
            max_goals=args.max_goals,
        )
        rows.append(
            {
                "alpha_on_dixon_coles": float(alpha_on_dixon_coles),
                "alpha_on_elo_multinomial": float(1.0 - float(alpha_on_dixon_coles)),
                "matches": match_rows,
                "expected_points": expected_points,
                "group_winner_probability": monte_carlo["group_winner_probability"],
            }
        )

    output = {
        "group": group_name,
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
    elo_benchmark: EloMultinomialBenchmark,
    group_schedule: pd.DataFrame,
    elo_ratings: dict[str, float],
    alpha_on_dixon_coles: float,
    max_goals: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    teams = sorted(set(group_schedule["home_team"]).union(group_schedule["away_team"]))
    expected_points = {team: 0.0 for team in teams}
    match_rows: list[dict[str, object]] = []

    for row in group_schedule.itertuples(index=False):
        probabilities = _hybrid_probabilities(
            model=model,
            elo_benchmark=elo_benchmark,
            home_team=str(row.home_team),
            away_team=str(row.away_team),
            neutral=bool(True),
            elo_diff=float(elo_ratings.get(str(row.home_team), 1500.0) - elo_ratings.get(str(row.away_team), 1500.0)),
            alpha_on_dixon_coles=alpha_on_dixon_coles,
            max_goals=max_goals,
        )
        match_rows.append(
            {
                "home_team": str(row.home_team),
                "away_team": str(row.away_team),
                "home_win_probability": float(probabilities.home),
                "draw_probability": float(probabilities.draw),
                "away_win_probability": float(probabilities.away),
            }
        )
        expected_points[str(row.home_team)] += (3.0 * float(probabilities.home)) + float(probabilities.draw)
        expected_points[str(row.away_team)] += (3.0 * float(probabilities.away)) + float(probabilities.draw)

    return match_rows, expected_points


def _simulate_group_points(
    *,
    model: object,
    elo_benchmark: EloMultinomialBenchmark,
    group_schedule: pd.DataFrame,
    elo_ratings: dict[str, float],
    alpha_on_dixon_coles: float,
    simulations: int,
    random_state: int,
    max_goals: int,
) -> dict[str, dict[str, float]]:
    teams = sorted(set(group_schedule["home_team"]).union(group_schedule["away_team"]))
    group_winner_total = {team: 0 for team in teams}
    rng = np.random.default_rng(random_state)

    for _ in range(simulations):
        simulated_matches = []
        for row in group_schedule.itertuples(index=False):
            home_team = str(row.home_team)
            away_team = str(row.away_team)
            elo_diff = float(elo_ratings.get(home_team, 1500.0) - elo_ratings.get(away_team, 1500.0))
            baseline_matrix = model.predict_score_matrix(
                home_team,
                away_team,
                neutral_site=True,
                elo_diff_pre=elo_diff,
                max_goals=max_goals,
            )
            hybrid_probabilities = _hybrid_probabilities(
                model=model,
                elo_benchmark=elo_benchmark,
                home_team=home_team,
                away_team=away_team,
                neutral=True,
                elo_diff=elo_diff,
                alpha_on_dixon_coles=alpha_on_dixon_coles,
                max_goals=max_goals,
            )
            hybrid_matrix = reweight_score_matrix_to_outcomes(
                baseline_matrix,
                target_probabilities=hybrid_probabilities,
            )
            home_goals, away_goals = sample_scoreline_from_matrix(hybrid_matrix, rng)
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
            if int(standing_row.group_rank) == 1:
                group_winner_total[str(standing_row.team)] += 1

    return {
        "group_winner_probability": {
            team: group_winner_total[team] / simulations
            for team in teams
        }
    }


def _hybrid_probabilities(
    *,
    model: object,
    elo_benchmark: EloMultinomialBenchmark,
    home_team: str,
    away_team: str,
    neutral: bool,
    elo_diff: float,
    alpha_on_dixon_coles: float,
    max_goals: int,
):
    baseline_probabilities = model.predict_outcome_probabilities(
        home_team,
        away_team,
        neutral_site=neutral,
        elo_diff_pre=elo_diff,
        max_goals=max_goals,
    )
    elo_probabilities = elo_benchmark.predict_proba(
        type(
            "HybridRow",
            (),
            {
                "elo_diff_pre": float(elo_diff),
                "neutral": bool(neutral),
            },
        )()
    )
    return blend_three_way_probabilities(
        baseline_probabilities,
        elo_probabilities,
        alpha_on_base=alpha_on_dixon_coles,
    )


if __name__ == "__main__":
    main()
