from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.data import canonicalize_team_name, load_international_results
from wc2026_model.evaluation import EloMultinomialBenchmark
from wc2026_model.evaluation.scoring import (
    brier_score_three_way,
    log_loss_three_way,
    ranked_probability_score,
)
from wc2026_model.features import augment_with_pre_match_elo
from wc2026_model.models import blend_three_way_probabilities, power_calibrate_probabilities
from wc2026_model.pipeline import BaselineTrainingConfig, train_baseline_model


def _parse_years(value: str) -> list[int]:
    years = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not years:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated year.")
    return years


def _parse_float_grid(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float value.")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate squad-strength scaling on historical World Cup matches by training "
            "pre-tournament models and sweeping the squad adjustment."
        )
    )
    parser.add_argument(
        "--results-input",
        default="data/interim/international_results_augmented.csv",
    )
    parser.add_argument(
        "--squad-strength-input",
        default="reports/historical_world_cup_squad_strength_ratings.csv",
    )
    parser.add_argument(
        "--rating-column",
        default="squad_club_elo_rating",
        help="Team-strength column to convert into Elo-like adjustments.",
    )
    parser.add_argument(
        "--secondary-rating-column",
        default=None,
        help=(
            "Optional second team-strength column to add as a separate Elo-like adjustment."
        ),
    )
    parser.add_argument("--years", type=_parse_years, default=[2014, 2018, 2022])
    parser.add_argument(
        "--scale-grid",
        type=_parse_float_grid,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--secondary-scale-grid",
        type=_parse_float_grid,
        default=None,
        help=(
            "Optional grid for the secondary rating-column adjustment. "
            "Defaults to the primary scale grid when --secondary-rating-column is provided."
        ),
    )
    parser.add_argument("--min-match-date", default="2010-01-01")
    parser.add_argument("--min-team-matches", type=int, default=10)
    parser.add_argument("--time-decay-xi", type=float, default=0.001)
    parser.add_argument("--l2-penalty", type=float, default=0.01)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--hybrid-alpha", type=float, default=0.75)
    parser.add_argument("--gamma-home", type=float, default=1.05)
    parser.add_argument("--gamma-draw", type=float, default=1.0)
    parser.add_argument("--gamma-away", type=float, default=1.1)
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument(
        "--predictions-output",
        default="reports/historical_world_cup_squad_scale_predictions.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/historical_world_cup_squad_scale_summary.csv",
    )
    parser.add_argument(
        "--aggregate-output",
        default="reports/historical_world_cup_squad_scale_aggregate.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    results = load_international_results(args.results_input)
    results["match_date"] = pd.to_datetime(results["match_date"], errors="raise")
    results = results.sort_values(["match_date", "home_team", "away_team"], kind="stable").reset_index(
        drop=True
    )
    squad_strengths = pd.read_csv(args.squad_strength_input)

    config = BaselineTrainingConfig(
        min_match_date=args.min_match_date,
        min_team_matches=args.min_team_matches,
        time_decay_xi=args.time_decay_xi,
        l2_penalty=args.l2_penalty,
        maxiter=args.maxiter,
    )
    full_with_elo = augment_with_pre_match_elo(results, config=config.elo_config)

    prediction_rows: list[dict[str, object]] = []
    tournament_rows: list[dict[str, object]] = []
    secondary_scale_grid = (
        args.secondary_scale_grid
        if args.secondary_rating_column and args.secondary_scale_grid is not None
        else (args.scale_grid if args.secondary_rating_column else [0.0])
    )

    for year in args.years:
        tournament_frame = full_with_elo[
            (full_with_elo["tournament"] == "FIFA World Cup")
            & (full_with_elo["match_date"].dt.year == year)
        ].copy()
        if tournament_frame.empty:
            continue

        tournament_start = pd.Timestamp(tournament_frame["match_date"].min())
        model, training_frame = train_baseline_model(
            results,
            config=BaselineTrainingConfig(
                min_match_date=args.min_match_date,
                training_cutoff=tournament_start.strftime("%Y-%m-%d"),
                min_team_matches=args.min_team_matches,
                time_decay_xi=args.time_decay_xi,
                l2_penalty=args.l2_penalty,
                maxiter=args.maxiter,
                elo_config=config.elo_config,
            ),
        )
        tournament_frame = tournament_frame[
            tournament_frame["home_team"].isin(model.teams) & tournament_frame["away_team"].isin(model.teams)
        ].reset_index(drop=True)
        if tournament_frame.empty or training_frame.empty:
            continue

        elo_benchmark = EloMultinomialBenchmark.fit(training_frame)
        team_strength_lookup = _build_team_strength_lookup(
            squad_strengths,
            year=year,
            rating_column=args.rating_column,
        )
        secondary_team_strength_lookup = (
            _build_team_strength_lookup(
                squad_strengths,
                year=year,
                rating_column=args.secondary_rating_column,
            )
            if args.secondary_rating_column
            else {}
        )

        for scale in args.scale_grid:
            for secondary_scale in secondary_scale_grid:
                scale_prediction_rows = []
                for row in tournament_frame.itertuples(index=False):
                    adjusted_elo_diff = _apply_team_strength_adjustments(
                        base_elo_diff=float(row.elo_diff_pre),
                        home_team=str(row.home_team),
                        away_team=str(row.away_team),
                        primary_lookup=team_strength_lookup,
                        primary_scale=float(scale),
                        secondary_lookup=secondary_team_strength_lookup,
                        secondary_scale=float(secondary_scale),
                    )
                    base_probabilities = model.predict_outcome_probabilities(
                        row.home_team,
                        row.away_team,
                        neutral_site=bool(row.neutral),
                        elo_diff_pre=adjusted_elo_diff,
                        max_goals=args.max_goals,
                    )
                    elo_row = _ScaledEloRow(
                        elo_diff_pre=adjusted_elo_diff,
                        neutral=bool(row.neutral),
                    )
                    blended = blend_three_way_probabilities(
                        base_probabilities,
                        elo_benchmark.predict_proba(elo_row),
                        alpha_on_base=args.hybrid_alpha,
                    )
                    probabilities = power_calibrate_probabilities(
                        blended,
                        gamma_home=args.gamma_home,
                        gamma_draw=args.gamma_draw,
                        gamma_away=args.gamma_away,
                    )
                    prediction_row = {
                        "model_name": _build_model_name(
                            primary_rating_column=args.rating_column,
                            primary_scale=float(scale),
                            secondary_rating_column=args.secondary_rating_column,
                            secondary_scale=float(secondary_scale),
                        ),
                        "tournament_year": year,
                        "match_date": row.match_date,
                        "home_team": row.home_team,
                        "away_team": row.away_team,
                        "actual_outcome": row.home_result,
                        "neutral": bool(row.neutral),
                        "elo_diff_pre": float(row.elo_diff_pre),
                        "adjusted_elo_diff_pre": adjusted_elo_diff,
                        "primary_scale": float(scale),
                        "secondary_scale": float(secondary_scale),
                        "home_team_squad_strength": team_strength_lookup.get(str(row.home_team)),
                        "away_team_squad_strength": team_strength_lookup.get(str(row.away_team)),
                        "home_team_secondary_strength": (
                            secondary_team_strength_lookup.get(str(row.home_team))
                            if args.secondary_rating_column
                            else None
                        ),
                        "away_team_secondary_strength": (
                            secondary_team_strength_lookup.get(str(row.away_team))
                            if args.secondary_rating_column
                            else None
                        ),
                        "pred_home": probabilities.home,
                        "pred_draw": probabilities.draw,
                        "pred_away": probabilities.away,
                        "log_loss": log_loss_three_way(probabilities, str(row.home_result)),
                        "brier_score": brier_score_three_way(probabilities, str(row.home_result)),
                        "ranked_probability_score": ranked_probability_score(
                            probabilities, str(row.home_result)
                        ),
                    }
                    prediction_rows.append(prediction_row)
                    scale_prediction_rows.append(prediction_row)

                if scale_prediction_rows:
                    scale_frame = pd.DataFrame(scale_prediction_rows)
                    tournament_rows.append(
                        {
                            "tournament_year": year,
                            "primary_scale": float(scale),
                            "secondary_scale": float(secondary_scale),
                            "secondary_rating_column": args.secondary_rating_column,
                            "matches": int(len(scale_frame)),
                            "mean_log_loss": float(scale_frame["log_loss"].mean()),
                            "mean_brier_score": float(scale_frame["brier_score"].mean()),
                            "mean_ranked_probability_score": float(
                                scale_frame["ranked_probability_score"].mean()
                            ),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    tournament_summary = pd.DataFrame(tournament_rows).sort_values(
        ["mean_log_loss", "mean_brier_score", "mean_ranked_probability_score"],
        ascending=[True, True, True],
        kind="stable",
    )
    aggregate_summary = _summarize_scale_predictions(predictions)

    best_overall = None
    if not aggregate_summary.empty:
        best_overall = aggregate_summary.sort_values("mean_log_loss", kind="stable").iloc[0].to_dict()

    aggregate_output = {
        "years": args.years,
        "scale_grid": args.scale_grid,
        "rating_column": args.rating_column,
        "secondary_rating_column": args.secondary_rating_column,
        "secondary_scale_grid": secondary_scale_grid if args.secondary_rating_column else [],
        "best_overall": best_overall,
        "tournament_best_rows": tournament_summary.groupby("tournament_year", sort=True)
        .head(1)
        .to_dict(orient="records"),
    }

    predictions_output = Path(args.predictions_output)
    summary_output = Path(args.summary_output)
    aggregate_output_path = Path(args.aggregate_output)
    for path in (predictions_output, summary_output, aggregate_output_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(predictions_output, index=False)
    tournament_summary.to_csv(summary_output, index=False)
    aggregate_output_path.write_text(json.dumps(aggregate_output, indent=2), encoding="utf-8")

    print(json.dumps(aggregate_output, indent=2))


class _ScaledEloRow:
    def __init__(self, *, elo_diff_pre: float, neutral: bool) -> None:
        self.elo_diff_pre = elo_diff_pre
        self.neutral = neutral


def _build_team_strength_lookup(
    squad_strengths: pd.DataFrame,
    *,
    year: int,
    rating_column: str = "squad_club_elo_rating",
) -> dict[str, float]:
    if rating_column not in squad_strengths.columns:
        raise ValueError(
            f"Squad-strength frame is missing rating column '{rating_column}'."
        )
    year_frame = squad_strengths.loc[squad_strengths["tournament_year"] == year].copy()
    if year_frame.empty:
        return {}
    return {
        canonicalize_team_name(str(row.team)): float(getattr(row, rating_column))
        for row in year_frame.loc[:, ["team", rating_column]].itertuples(index=False)
        if pd.notna(getattr(row, rating_column))
    }


def _adjust_elo_diff_for_squad_strength(
    *,
    base_elo_diff: float,
    home_team: str,
    away_team: str,
    team_strength_lookup: dict[str, float],
    scale: float,
) -> float:
    home_strength = float(team_strength_lookup.get(canonicalize_team_name(home_team), 0.0))
    away_strength = float(team_strength_lookup.get(canonicalize_team_name(away_team), 0.0))
    return float(base_elo_diff) + (float(scale) * (home_strength - away_strength))


def _apply_team_strength_adjustments(
    *,
    base_elo_diff: float,
    home_team: str,
    away_team: str,
    primary_lookup: dict[str, float],
    primary_scale: float,
    secondary_lookup: dict[str, float] | None = None,
    secondary_scale: float = 0.0,
) -> float:
    adjusted = _adjust_elo_diff_for_squad_strength(
        base_elo_diff=base_elo_diff,
        home_team=home_team,
        away_team=away_team,
        team_strength_lookup=primary_lookup,
        scale=primary_scale,
    )
    if secondary_lookup:
        adjusted = _adjust_elo_diff_for_squad_strength(
            base_elo_diff=adjusted,
            home_team=home_team,
            away_team=away_team,
            team_strength_lookup=secondary_lookup,
            scale=secondary_scale,
        )
    return adjusted


def _build_model_name(
    *,
    primary_rating_column: str,
    primary_scale: float,
    secondary_rating_column: str | None = None,
    secondary_scale: float = 0.0,
) -> str:
    model_name = f"{primary_rating_column}__scale_{primary_scale:g}"
    if secondary_rating_column:
        model_name += f"__{secondary_rating_column}__scale_{secondary_scale:g}"
    return model_name


def _summarize_scale_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "predictions",
                "mean_log_loss",
                "mean_brier_score",
                "mean_ranked_probability_score",
            ]
        )

    rows: list[dict[str, object]] = []
    for model_name, model_predictions in predictions.groupby("model_name", sort=True):
        rows.append(
            {
                "model_name": model_name,
                "predictions": int(len(model_predictions)),
                "mean_log_loss": float(model_predictions["log_loss"].mean()),
                "mean_brier_score": float(model_predictions["brier_score"].mean()),
                "mean_ranked_probability_score": float(
                    model_predictions["ranked_probability_score"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_log_loss", kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    main()
