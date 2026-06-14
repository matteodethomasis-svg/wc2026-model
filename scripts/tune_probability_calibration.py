from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from wc2026_model.evaluation.calibration import (
    power_calibrate_prediction_frame,
)
from wc2026_model.evaluation.scoring import (
    brier_score_three_way,
    log_loss_three_way,
    ranked_probability_score,
)
from wc2026_model.types import ThreeWayProbabilities


def _parse_float_grid(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float value.")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune outcome-specific power calibration on out-of-sample three-way predictions."
    )
    parser.add_argument(
        "--predictions-input",
        default="reports/benchmark_backtest_predictions.csv",
        help="Prediction CSV containing pred_home, pred_draw, pred_away and actual_outcome.",
    )
    parser.add_argument(
        "--model-name",
        default="dixon_coles_elo",
        help="Model name to filter inside the predictions CSV.",
    )
    parser.add_argument(
        "--calibrated-model-name",
        default=None,
        help="Optional model_name assigned to the calibrated predictions output. Defaults to <model-name>_calibrated.",
    )
    parser.add_argument(
        "--gamma-home-grid",
        type=_parse_float_grid,
        default=[0.9, 1.0, 1.1, 1.2],
        help="Comma-separated grid for home probability power.",
    )
    parser.add_argument(
        "--gamma-draw-grid",
        type=_parse_float_grid,
        default=[0.8, 0.9, 1.0, 1.1],
        help="Comma-separated grid for draw probability power.",
    )
    parser.add_argument(
        "--gamma-away-grid",
        type=_parse_float_grid,
        default=[0.9, 1.0, 1.1, 1.2],
        help="Comma-separated grid for away probability power.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/probability_calibration_tuning_summary.csv",
    )
    parser.add_argument(
        "--best-output",
        default="reports/probability_calibration_tuning_best.json",
    )
    parser.add_argument(
        "--predictions-output",
        default=None,
        help="Optional path to save the original predictions plus an appended calibrated model block using the best gammas.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    predictions = pd.read_csv(args.predictions_input)
    filtered_predictions = predictions.loc[predictions["model_name"] == args.model_name].copy()
    if filtered_predictions.empty:
        raise ValueError(f"No predictions found for model_name={args.model_name!r}.")

    rows: list[dict[str, float | str]] = []
    for gamma_home, gamma_draw, gamma_away in itertools.product(
        args.gamma_home_grid,
        args.gamma_draw_grid,
        args.gamma_away_grid,
    ):
        calibrated = power_calibrate_prediction_frame(
            filtered_predictions,
            gamma_home=gamma_home,
            gamma_draw=gamma_draw,
            gamma_away=gamma_away,
        )
        metric_frame = _score_prediction_frame(calibrated)
        rows.append(
            {
                "model_name": args.model_name,
                "gamma_home": float(gamma_home),
                "gamma_draw": float(gamma_draw),
                "gamma_away": float(gamma_away),
                "mean_log_loss": float(metric_frame["log_loss"].mean()),
                "mean_brier_score": float(metric_frame["brier_score"].mean()),
                "mean_ranked_probability_score": float(
                    metric_frame["ranked_probability_score"].mean()
                ),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["mean_log_loss", "mean_brier_score", "mean_ranked_probability_score"],
        ascending=[True, True, True],
        kind="stable",
    )
    summary_output = Path(args.summary_output)
    best_output = Path(args.best_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    best_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)
    best_row = summary.iloc[0].to_dict()
    best_output.write_text(json.dumps(best_row, indent=2), encoding="utf-8")

    predictions_output = args.predictions_output
    calibrated_model_name = args.calibrated_model_name or f"{args.model_name}_calibrated"
    if predictions_output is not None:
        best_predictions = power_calibrate_prediction_frame(
            filtered_predictions,
            gamma_home=float(best_row["gamma_home"]),
            gamma_draw=float(best_row["gamma_draw"]),
            gamma_away=float(best_row["gamma_away"]),
        )
        best_predictions["model_name"] = calibrated_model_name
        augmented_predictions = pd.concat(
            [predictions, best_predictions],
            ignore_index=True,
        )
        predictions_output_path = Path(predictions_output)
        predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
        augmented_predictions.to_csv(predictions_output_path, index=False)

    print(json.dumps({"best_configuration": best_row}, indent=2))
    print(f"Saved probability calibration sweep to {summary_output}")
    if predictions_output is not None:
        print(f"Saved augmented predictions with calibrated rows to {predictions_output}")


def _score_prediction_frame(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in prediction_frame.itertuples(index=False):
        probabilities = ThreeWayProbabilities(
            home=float(row.pred_home),
            draw=float(row.pred_draw),
            away=float(row.pred_away),
        )
        rows.append(
            {
                "log_loss": log_loss_three_way(probabilities, str(row.actual_outcome)),
                "brier_score": brier_score_three_way(probabilities, str(row.actual_outcome)),
                "ranked_probability_score": ranked_probability_score(
                    probabilities,
                    str(row.actual_outcome),
                ),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
