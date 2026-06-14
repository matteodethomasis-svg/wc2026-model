from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from wc2026_model.data import download_international_results_csv, load_international_results
from wc2026_model.features import build_latest_elo_ratings
from wc2026_model.pipeline import BaselineTrainingConfig, train_baseline_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the baseline World Cup 2026 pre-match model."
    )
    parser.add_argument(
        "--input",
        default="data/raw/international_results.csv",
        help="Path to the international results CSV.",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Download the input CSV automatically if it does not exist.",
    )
    parser.add_argument(
        "--min-match-date",
        default="2010-01-01",
        help="Earliest match date to include in training.",
    )
    parser.add_argument(
        "--training-cutoff",
        default=None,
        help="Exclusive upper bound for training matches (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--min-team-matches",
        type=int,
        default=10,
        help="Minimum number of matches a team must have to be retained.",
    )
    parser.add_argument(
        "--time-decay-xi",
        type=float,
        default=0.001,
        help="Exponential time-decay coefficient.",
    )
    parser.add_argument(
        "--l2-penalty",
        type=float,
        default=0.01,
        help="L2 penalty used in Dixon-Coles training.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum number of optimizer iterations.",
    )
    parser.add_argument(
        "--model-output",
        default="models/baseline_dixon_coles_elo.pkl",
        help="Path to save the fitted model pickle.",
    )
    parser.add_argument(
        "--team-strengths-output",
        default="reports/baseline_team_strengths.csv",
        help="Path to save the estimated team strengths CSV.",
    )
    parser.add_argument(
        "--training-frame-output",
        default="reports/baseline_training_frame.csv",
        help="Path to save the final training frame CSV.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/baseline_fit_summary.json",
        help="Path to save the fit summary JSON.",
    )
    parser.add_argument(
        "--elo-ratings-output",
        default="reports/baseline_latest_elo_ratings.csv",
        help="Path to save the latest Elo ratings CSV.",
    )
    return parser


def _ensure_input_exists(input_path: Path, auto_download: bool) -> Path:
    if input_path.exists():
        return input_path
    if not auto_download:
        raise FileNotFoundError(
            f"Input CSV not found at {input_path}. Re-run with --auto-download or provide --input."
        )
    return download_international_results_csv(input_path)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = _ensure_input_exists(Path(args.input), args.auto_download)
    results = load_international_results(input_path)

    config = BaselineTrainingConfig(
        min_match_date=args.min_match_date,
        training_cutoff=args.training_cutoff,
        min_team_matches=args.min_team_matches,
        time_decay_xi=args.time_decay_xi,
        l2_penalty=args.l2_penalty,
        maxiter=args.maxiter,
    )
    model, training_frame = train_baseline_model(results, config=config)

    model_output = Path(args.model_output)
    team_strengths_output = Path(args.team_strengths_output)
    training_frame_output = Path(args.training_frame_output)
    summary_output = Path(args.summary_output)
    elo_ratings_output = Path(args.elo_ratings_output)

    for path in (
        model_output,
        team_strengths_output,
        training_frame_output,
        summary_output,
        elo_ratings_output,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    with model_output.open("wb") as file_handle:
        pickle.dump(model, file_handle)

    model.team_strengths().to_csv(team_strengths_output, index=False)
    training_frame.to_csv(training_frame_output, index=False)
    elo_ratings = build_latest_elo_ratings(training_frame, config=config.elo_config)
    elo_ratings.to_csv(elo_ratings_output, index=False)

    summary = {
        "fit_success": model.fit_result.success,
        "fit_message": model.fit_result.message,
        "iterations": model.fit_result.iterations,
        "objective_value": model.fit_result.objective_value,
        "team_count": len(model.teams),
        "match_count": int(len(training_frame)),
        "training_cutoff": args.training_cutoff,
        "min_match_date": args.min_match_date,
        "time_decay_xi": args.time_decay_xi,
        "l2_penalty": args.l2_penalty,
        "elo_ratings_output": str(elo_ratings_output),
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved model to {model_output}")
    print(f"Saved team strengths to {team_strengths_output}")
    print(f"Saved training frame to {training_frame_output}")
    print(f"Saved latest Elo ratings to {elo_ratings_output}")


if __name__ == "__main__":
    main()
