from __future__ import annotations

import argparse
import json
from pathlib import Path

from wc2026_model.data import download_international_results_csv, load_international_results
from wc2026_model.evaluation import (
    generate_rolling_cutoffs,
    run_expanding_window_backtest,
    summarize_backtest_folds,
)
from wc2026_model.pipeline import BaselineTrainingConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an expanding-window backtest for the baseline World Cup model."
    )
    parser.add_argument("--input", default="data/raw/international_results.csv")
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument("--min-match-date", default="2010-01-01")
    parser.add_argument("--backtest-start", default="2018-01-01")
    parser.add_argument("--backtest-end", default=None)
    parser.add_argument("--step-days", type=int, default=180)
    parser.add_argument("--test-window-days", type=int, default=120)
    parser.add_argument("--min-team-matches", type=int, default=10)
    parser.add_argument("--time-decay-xi", type=float, default=0.001)
    parser.add_argument("--l2-penalty", type=float, default=0.01)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument(
        "--predictions-output",
        default="reports/backtest_predictions.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/backtest_summary.csv",
    )
    parser.add_argument(
        "--aggregate-output",
        default="reports/backtest_aggregate.json",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        if not args.auto_download:
            raise FileNotFoundError(
                f"Input CSV not found at {input_path}. Use --auto-download or provide --input."
            )
        download_international_results_csv(input_path)

    results = load_international_results(input_path)
    config = BaselineTrainingConfig(
        min_match_date=args.min_match_date,
        min_team_matches=args.min_team_matches,
        time_decay_xi=args.time_decay_xi,
        l2_penalty=args.l2_penalty,
        maxiter=args.maxiter,
    )
    cutoffs = generate_rolling_cutoffs(
        results,
        start_date=args.backtest_start,
        end_date=args.backtest_end,
        step_days=args.step_days,
    )
    predictions, fold_summaries = run_expanding_window_backtest(
        results,
        config=config,
        cutoffs=cutoffs,
        test_window_days=args.test_window_days,
    )
    summary_frame = summarize_backtest_folds(fold_summaries)

    predictions_output = Path(args.predictions_output)
    summary_output = Path(args.summary_output)
    aggregate_output = Path(args.aggregate_output)
    for path in (predictions_output, summary_output, aggregate_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(predictions_output, index=False)
    summary_frame.to_csv(summary_output, index=False)

    aggregate = {
        "fold_count": int(len(summary_frame)),
        "total_predictions": int(len(predictions)),
        "mean_log_loss": float(summary_frame["average_log_loss"].mean()),
        "mean_brier_score": float(summary_frame["average_brier_score"].mean()),
        "mean_ranked_probability_score": float(
            summary_frame["average_ranked_probability_score"].mean()
        ),
    }
    aggregate_output.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
