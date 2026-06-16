from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from wc2026_model.data import canonicalize_team_name
from wc2026_model.evaluation import EloMultinomialBenchmark
from wc2026_model.models import BlendedMatchModel, CalibratedMatchModel
from wc2026_model.tournament import load_played_group_results, simulate_world_cup_2026


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate the FIFA World Cup 2026 using the fitted baseline model."
    )
    parser.add_argument(
        "--model-input",
        default="models/baseline_dixon_coles_elo.pkl",
        help="Path to the fitted model pickle.",
    )
    parser.add_argument(
        "--groups-input",
        required=True,
        help="CSV containing World Cup groups with columns group, team and optional slot.",
    )
    parser.add_argument(
        "--elo-ratings-input",
        default="reports/baseline_latest_elo_ratings.csv",
        help="CSV containing team-level Elo ratings with columns team and elo_rating.",
    )
    parser.add_argument(
        "--training-frame-input",
        default="reports/baseline_training_frame.csv",
        help="CSV containing the baseline training frame for the optional Elo benchmark blend.",
    )
    parser.add_argument(
        "--elo-blend-alpha",
        type=float,
        default=1.0,
        help="Weight on the baseline Dixon-Coles probabilities. Set below 1.0 to blend with Elo multinomial probabilities.",
    )
    parser.add_argument(
        "--blend-method", choices=["linear", "log_pool"], default="linear",
        help="DC/Elo combine rule: linear or log_pool (validated ~0.5%% better log loss).",
    )
    parser.add_argument(
        "--blend-temperature", type=float, default=1.0,
        help="Temperature for log_pool blending. Ignored for linear.",
    )
    parser.add_argument(
        "--calibration-gamma-home",
        type=float,
        default=1.0,
        help="Outcome-specific power calibration for home-win probability.",
    )
    parser.add_argument(
        "--calibration-gamma-draw",
        type=float,
        default=1.0,
        help="Outcome-specific power calibration for draw probability.",
    )
    parser.add_argument(
        "--calibration-gamma-away",
        type=float,
        default=1.0,
        help="Outcome-specific power calibration for away-win probability.",
    )
    parser.add_argument(
        "--squad-strength-input",
        default=None,
        help="Optional CSV containing team-level squad strength ratings.",
    )
    parser.add_argument(
        "--squad-strength-column",
        default="squad_club_elo_rating",
        help="Column in --squad-strength-input used as the squad rating.",
    )
    parser.add_argument(
        "--secondary-squad-strength-column",
        default=None,
        help="Optional second squad-strength column used as an additional adjustment.",
    )
    parser.add_argument(
        "--squad-elo-scale",
        type=float,
        default=0.0,
        help="Adds scale * squad strength to team Elo ratings during simulation.",
    )
    parser.add_argument(
        "--secondary-squad-elo-scale",
        type=float,
        default=0.0,
        help="Adds scale * secondary squad strength to team Elo ratings during simulation.",
    )
    parser.add_argument(
        "--elo-temperature",
        type=float,
        default=1.0,
        help=(
            "Softens Elo gaps inside the tournament simulation (elo_diff / T). T>1 adds "
            "upset variance so favourites don't compound into over-confident title odds. "
            "T=1 keeps current behaviour."
        ),
    )
    parser.add_argument(
        "--results-input",
        default=None,
        help="Optional standardized or raw results CSV used to condition on matches already played.",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Optional inclusive cutoff date (YYYY-MM-DD) used with --results-input.",
    )
    parser.add_argument(
        "--tournament-year",
        type=int,
        default=2026,
        help="Calendar year of the tournament used to exclude historical World Cup matches with the same pairings.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="Number of tournament simulations to run.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=2026,
        help="Random seed used for the Monte Carlo simulation.",
    )
    parser.add_argument(
        "--output",
        default="reports/wc2026_simulation_probabilities.csv",
        help="Path to save the team probability table.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/wc2026_simulation_summary.json",
        help="Path to save a compact simulation summary JSON.",
    )
    return parser


def _load_groups(path: Path) -> dict[str, list[str]]:
    groups_frame = pd.read_csv(path)
    required_columns = {"group", "team"}
    missing_columns = required_columns.difference(groups_frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in groups CSV: {missing}")

    groups_frame = groups_frame.copy()
    groups_frame["group"] = groups_frame["group"].astype(str)
    groups_frame["team"] = groups_frame["team"].astype(str).map(canonicalize_team_name)
    if "slot" in groups_frame.columns:
        groups_frame = groups_frame.sort_values(["group", "slot"], kind="stable")
    else:
        groups_frame["_row_order"] = range(len(groups_frame))
        groups_frame = groups_frame.sort_values(["group", "_row_order"], kind="stable")

    grouped = groups_frame.groupby("group", sort=True)["team"].apply(list)
    return grouped.to_dict()


def _load_elo_ratings(path: Path) -> dict[str, float]:
    ratings_frame = pd.read_csv(path)
    required_columns = {"team", "elo_rating"}
    missing_columns = required_columns.difference(ratings_frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in Elo CSV: {missing}")

    return {
        canonicalize_team_name(str(row.team)): float(row.elo_rating)
        for row in ratings_frame.loc[:, ["team", "elo_rating"]].itertuples(index=False)
    }


def _load_team_strength_ratings(
    path: Path | None,
    *,
    rating_column: str,
) -> dict[str, float]:
    if path is None:
        return {}
    ratings_frame = pd.read_csv(path)
    required_columns = {"team", rating_column}
    missing_columns = required_columns.difference(ratings_frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in squad strength CSV: {missing}")

    return {
        canonicalize_team_name(str(row.team)): float(getattr(row, rating_column))
        for row in ratings_frame.loc[:, ["team", rating_column]].itertuples(index=False)
        if pd.notna(getattr(row, rating_column))
    }


def _build_model_team_lookup(model: object) -> dict[str, str]:
    teams = getattr(model, "teams", None)
    if teams is None:
        return {}
    return {canonicalize_team_name(str(team)): str(team) for team in teams}


def _align_groups_to_model(
    groups: dict[str, list[str]],
    *,
    model_team_lookup: dict[str, str],
) -> dict[str, list[str]]:
    return {
        group_name: [model_team_lookup.get(team, team) for team in team_list]
        for group_name, team_list in groups.items()
    }


def _align_elo_ratings_to_model(
    elo_ratings: dict[str, float],
    *,
    model_team_lookup: dict[str, str],
) -> dict[str, float]:
    aligned: dict[str, float] = {}
    for team, elo_rating in elo_ratings.items():
        aligned[model_team_lookup.get(team, team)] = float(elo_rating)
    return aligned


def _align_played_group_results_to_model(
    played_group_results: pd.DataFrame | None,
    *,
    model_team_lookup: dict[str, str],
) -> pd.DataFrame | None:
    if played_group_results is None or played_group_results.empty:
        return played_group_results

    aligned = played_group_results.copy()
    aligned["home_team"] = aligned["home_team"].map(lambda team: model_team_lookup.get(team, team))
    aligned["away_team"] = aligned["away_team"].map(lambda team: model_team_lookup.get(team, team))
    return aligned


def _align_team_strength_ratings_to_model(
    team_strength_ratings: dict[str, float],
    *,
    model_team_lookup: dict[str, str],
) -> dict[str, float]:
    aligned: dict[str, float] = {}
    for team, rating in team_strength_ratings.items():
        aligned[model_team_lookup.get(team, team)] = float(rating)
    return aligned


def _maybe_build_blended_model(
    model: object,
    *,
    training_frame_path: Path,
    alpha_on_base: float,
    blend_method: str = "linear",
    blend_temperature: float = 1.0,
) -> object:
    if alpha_on_base >= 1.0:
        return model
    training_frame = pd.read_csv(training_frame_path)
    elo_benchmark = EloMultinomialBenchmark.fit(training_frame)
    return BlendedMatchModel(
        base_model=model,
        overlay_model=elo_benchmark,
        alpha_on_base=alpha_on_base,
        blend_method=blend_method,
        blend_temperature=blend_temperature,
    )


def _maybe_build_calibrated_model(
    model: object,
    *,
    gamma_home: float,
    gamma_draw: float,
    gamma_away: float,
) -> object:
    if gamma_home == 1.0 and gamma_draw == 1.0 and gamma_away == 1.0:
        return model
    return CalibratedMatchModel(
        base_model=model,
        gamma_home=gamma_home,
        gamma_draw=gamma_draw,
        gamma_away=gamma_away,
    )


def _combine_elo_and_squad_strength(
    *,
    elo_ratings: dict[str, float],
    team_strength_ratings: dict[str, float],
    squad_elo_scale: float,
) -> dict[str, float]:
    return _combine_elo_and_squad_strength_layers(
        elo_ratings=elo_ratings,
        strength_layers=[(team_strength_ratings, squad_elo_scale)],
    )


def _combine_elo_and_squad_strength_layers(
    *,
    elo_ratings: dict[str, float],
    strength_layers: list[tuple[dict[str, float], float]],
) -> dict[str, float]:
    combined: dict[str, float] = {}
    teams = set(elo_ratings)
    for ratings, _ in strength_layers:
        teams.update(ratings)
    for team in teams:
        combined_rating = float(elo_ratings.get(team, 1500.0))
        for ratings, scale in strength_layers:
            combined_rating += float(scale) * float(ratings.get(team, 0.0))
        combined[team] = combined_rating
    return combined


def main() -> None:
    args = _build_parser().parse_args()

    model_path = Path(args.model_input)
    groups_path = Path(args.groups_input)
    elo_ratings_path = Path(args.elo_ratings_input)
    training_frame_path = Path(args.training_frame_input)
    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)
    squad_strength_path = Path(args.squad_strength_input) if args.squad_strength_input else None

    with model_path.open("rb") as file_handle:
        model = pickle.load(file_handle)
    model = _maybe_build_blended_model(
        model,
        training_frame_path=training_frame_path,
        alpha_on_base=args.elo_blend_alpha,
        blend_method=args.blend_method,
        blend_temperature=args.blend_temperature,
    )
    model = _maybe_build_calibrated_model(
        model,
        gamma_home=args.calibration_gamma_home,
        gamma_draw=args.calibration_gamma_draw,
        gamma_away=args.calibration_gamma_away,
    )

    groups = _load_groups(groups_path)
    elo_ratings = _load_elo_ratings(elo_ratings_path)
    team_strength_ratings = _load_team_strength_ratings(
        squad_strength_path,
        rating_column=args.squad_strength_column,
    )
    secondary_team_strength_ratings = (
        _load_team_strength_ratings(
            squad_strength_path,
            rating_column=args.secondary_squad_strength_column,
        )
        if args.secondary_squad_strength_column
        else {}
    )
    if (
        (args.squad_elo_scale != 0.0 and team_strength_ratings)
        or (args.secondary_squad_elo_scale != 0.0 and secondary_team_strength_ratings)
    ):
        elo_ratings = _combine_elo_and_squad_strength_layers(
            elo_ratings=elo_ratings,
            strength_layers=[
                (team_strength_ratings, args.squad_elo_scale),
                (secondary_team_strength_ratings, args.secondary_squad_elo_scale),
            ],
        )
    played_group_results = None
    if args.results_input is not None:
        played_group_results = load_played_group_results(
            Path(args.results_input),
            groups=groups,
            as_of_date=args.as_of_date,
            tournament_year=args.tournament_year,
        )

    model_team_lookup = _build_model_team_lookup(model)
    groups = _align_groups_to_model(groups, model_team_lookup=model_team_lookup)
    elo_ratings = _align_elo_ratings_to_model(
        elo_ratings,
        model_team_lookup=model_team_lookup,
    )
    team_strength_ratings = _align_team_strength_ratings_to_model(
        team_strength_ratings,
        model_team_lookup=model_team_lookup,
    )
    played_group_results = _align_played_group_results_to_model(
        played_group_results,
        model_team_lookup=model_team_lookup,
    )

    probabilities = simulate_world_cup_2026(
        model=model,
        groups=groups,
        elo_ratings=elo_ratings,
        played_group_results=played_group_results,
        simulations=args.simulations,
        random_state=args.random_state,
        elo_temperature=args.elo_temperature,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    probabilities.to_csv(output_path, index=False)

    summary = {
        "simulations": args.simulations,
        "random_state": args.random_state,
        "group_count": int(len(groups)),
        "team_count": int(sum(len(team_list) for team_list in groups.values())),
        "played_group_match_count": int(0 if played_group_results is None else len(played_group_results)),
        "as_of_date": args.as_of_date,
        "tournament_year": args.tournament_year,
        "elo_blend_alpha": args.elo_blend_alpha,
        "blend_method": args.blend_method,
        "blend_temperature": args.blend_temperature,
        "calibration_gamma_home": args.calibration_gamma_home,
        "calibration_gamma_draw": args.calibration_gamma_draw,
        "calibration_gamma_away": args.calibration_gamma_away,
        "squad_strength_column": args.squad_strength_column,
        "squad_elo_scale": args.squad_elo_scale,
        "secondary_squad_strength_column": args.secondary_squad_strength_column,
        "secondary_squad_elo_scale": args.secondary_squad_elo_scale,
        "teams_with_squad_strength_count": int(len(team_strength_ratings)),
        "teams_with_secondary_squad_strength_count": int(len(secondary_team_strength_ratings)),
        "top_outright_probabilities": probabilities.head(10).loc[
            :, ["team", "champion_probability", "reach_final_probability"]
        ].to_dict(orient="records"),
    }
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved simulation probabilities to {output_path}")
    print(f"Saved simulation summary to {summary_output_path}")


if __name__ == "__main__":
    main()
