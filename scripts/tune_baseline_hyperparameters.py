from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from wc2026_model.data import load_international_results
from wc2026_model.evaluation import (
    generate_rolling_cutoffs,
    run_expanding_window_backtest,
    summarize_backtest_folds,
)
from wc2026_model.pipeline import BaselineTrainingConfig


def _parse_float_grid(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float value.")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid-search baseline training hyperparameters on rolling backtests."
    )
    parser.add_argument(
        "--input",
        default="data/interim/international_results_augmented.csv",
        help="Path to the standardized or raw results CSV.",
    )
    parser.add_argument("--min-match-date", default="2010-01-01")
    parser.add_argument("--backtest-start", default="2025-06-01")
    parser.add_argument("--backtest-end", default=None)
    parser.add_argument("--step-days", type=int, default=90)
    parser.add_argument("--test-window-days", type=int, default=90)
    parser.add_argument("--min-team-matches", type=int, default=10)
    parser.add_argument(
        "--time-decay-grid",
        type=_parse_float_grid,
        default=[0.001, 0.0015, 0.002, 0.003],
        help="Comma-separated grid of time-decay xi values.",
    )
    parser.add_argument(
        "--l2-grid",
        type=_parse_float_grid,
        default=[0.01, 0.02],
        help="Comma-separated grid of L2 penalty values.",
    )
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument(
        "--summary-output",
        default="reports/hyperparameter_tuning_summary.csv",
        help="CSV summary for all tried configurations.",
    )
    parser.add_argument(
        "--best-output",
        default="reports/hyperparameter_tuning_best.json",
        help="JSON payload for the best configuration.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    results = load_international_results(Path(args.input))
    cutoffs = generate_rolling_cutoffs(
        results,
        start_date=args.backtest_start,
        end_date=args.backtest_end,
        step_days=args.step_days,
    )

    rows: list[dict[str, float | int | str]] = []
    for time_decay_xi, l2_penalty in itertools.product(args.time_decay_grid, args.l2_grid):
        config = BaselineTrainingConfig(
            min_match_date=args.min_match_date,
            min_team_matches=args.min_team_matches,
            time_decay_xi=time_decay_xi,
            l2_penalty=l2_penalty,
            maxiter=args.maxiter,
        )
        predictions, fold_summaries = run_expanding_window_backtest(
            results,
            config=config,
            cutoffs=cutoffs,
            test_window_days=args.test_window_days,
        )
        summary_frame = summarize_backtest_folds(fold_summaries)

        row = {
            "time_decay_xi": time_decay_xi,
            "l2_penalty": l2_penalty,
            "fold_count": int(len(summary_frame)),
            "predictions": int(len(predictions)),
            "mean_log_loss": float(summary_frame["average_log_loss"].mean()),
            "mean_brier_score": float(summary_frame["average_brier_score"].mean()),
            "mean_ranked_probability_score": float(
                summary_frame["average_ranked_probability_score"].mean()
            ),
        }
        rows.append(row)
        print(json.dumps(row))

    summary = pd.DataFrame.from_records(rows).sort_values(
        ["mean_log_loss", "mean_brier_score", "time_decay_xi", "l2_penalty"],
        ascending=[True, True, True, True],
        kind="stable",
    )

    summary_output = Path(args.summary_output)
    best_output = Path(args.best_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    best_output.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_output, index=False)
    best_row = summary.iloc[0].to_dict()
    best_output.write_text(json.dumps(best_row, indent=2), encoding="utf-8")

    print(f"Saved hyperparameter summary to {summary_output}")
    print(json.dumps({"best_configuration": best_row}, indent=2))


if __name__ == "__main__":
    main()
