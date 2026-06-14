from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc2026_model.evaluation import (
    build_convex_blend_predictions,
    expected_calibration_error_three_way,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append a blended benchmark model to an existing backtest prediction table."
    )
    parser.add_argument(
        "--predictions-input",
        default="reports/benchmark_backtest_predictions.csv",
    )
    parser.add_argument(
        "--base-model-name",
        default="dixon_coles_elo",
    )
    parser.add_argument(
        "--overlay-model-name",
        default="elo_multinomial",
    )
    parser.add_argument(
        "--blended-model-name",
        default="dixon_coles_elo_blend",
    )
    parser.add_argument(
        "--alpha-on-base",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--predictions-output",
        default="reports/benchmark_backtest_predictions_with_blend.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/benchmark_backtest_summary_with_blend.csv",
    )
    parser.add_argument(
        "--aggregate-output",
        default="reports/benchmark_backtest_aggregate_with_blend.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    predictions = pd.read_csv(args.predictions_input)
    blended_predictions = build_convex_blend_predictions(
        predictions,
        base_model_name=args.base_model_name,
        overlay_model_name=args.overlay_model_name,
        blended_model_name=args.blended_model_name,
        alpha_on_base=args.alpha_on_base,
    )
    combined_predictions = pd.concat([predictions, blended_predictions], ignore_index=True)
    summary = _summarize_predictions(combined_predictions)
    aggregate = _build_aggregate_report(summary)

    predictions_output = Path(args.predictions_output)
    summary_output = Path(args.summary_output)
    aggregate_output = Path(args.aggregate_output)
    for path in (predictions_output, summary_output, aggregate_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    combined_predictions.to_csv(predictions_output, index=False)
    summary.to_csv(summary_output, index=False)
    aggregate_output.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print(json.dumps(aggregate, indent=2))
    print(f"Saved blended predictions to {predictions_output}")


def _summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
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
                "expected_calibration_error": expected_calibration_error_three_way(
                    model_predictions
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_log_loss", kind="stable").reset_index(drop=True)


def _build_aggregate_report(summary: pd.DataFrame) -> dict[str, object]:
    rows = summary.to_dict(orient="records")
    best_row = rows[0]
    return {
        "best_model_by_log_loss": best_row["model_name"],
        "models": rows,
    }


if __name__ == "__main__":
    main()
