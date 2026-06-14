from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wc2026_model.evaluation.scoring import (
    brier_score_three_way,
    log_loss_three_way,
    ranked_probability_score,
)
from wc2026_model.types import ThreeWayProbabilities


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep convex blends between Dixon-Coles and Elo benchmark backtest predictions."
    )
    parser.add_argument(
        "--predictions-input",
        default="reports/benchmark_backtest_predictions.csv",
        help="Backtest predictions CSV produced by benchmark_backtest.py.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of equal alpha intervals between 0 and 1.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/dixon_coles_elo_blend_sweep.csv",
    )
    parser.add_argument(
        "--best-output",
        default="reports/dixon_coles_elo_blend_best.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    predictions = pd.read_csv(args.predictions_input)

    base = predictions.loc[predictions["model_name"] == "dixon_coles_elo"].copy()
    elo = predictions.loc[predictions["model_name"] == "elo_multinomial"].copy()
    merge_keys = ["cutoff_date", "match_date", "home_team", "away_team", "actual_outcome"]
    merged = base.merge(
        elo.loc[:, merge_keys + ["pred_home", "pred_draw", "pred_away"]],
        on=merge_keys,
        suffixes=("_dc", "_elo"),
        how="inner",
    )

    rows: list[dict[str, float]] = []
    for alpha_on_dixon_coles in np.linspace(0.0, 1.0, args.steps + 1):
        blended_home = (
            alpha_on_dixon_coles * merged["pred_home_dc"]
            + (1.0 - alpha_on_dixon_coles) * merged["pred_home_elo"]
        )
        blended_draw = (
            alpha_on_dixon_coles * merged["pred_draw_dc"]
            + (1.0 - alpha_on_dixon_coles) * merged["pred_draw_elo"]
        )
        blended_away = (
            alpha_on_dixon_coles * merged["pred_away_dc"]
            + (1.0 - alpha_on_dixon_coles) * merged["pred_away_elo"]
        )

        metric_rows = []
        for home, draw, away, actual_outcome in zip(
            blended_home,
            blended_draw,
            blended_away,
            merged["actual_outcome"],
        ):
            probabilities = ThreeWayProbabilities(
                home=float(home),
                draw=float(draw),
                away=float(away),
            )
            metric_rows.append(
                {
                    "log_loss": log_loss_three_way(probabilities, str(actual_outcome)),
                    "brier_score": brier_score_three_way(probabilities, str(actual_outcome)),
                    "ranked_probability_score": ranked_probability_score(
                        probabilities,
                        str(actual_outcome),
                    ),
                }
            )
        metric_frame = pd.DataFrame(metric_rows)
        rows.append(
            {
                "alpha_on_dixon_coles": float(alpha_on_dixon_coles),
                "alpha_on_elo_multinomial": float(1.0 - alpha_on_dixon_coles),
                "mean_log_loss": float(metric_frame["log_loss"].mean()),
                "mean_brier_score": float(metric_frame["brier_score"].mean()),
                "mean_ranked_probability_score": float(
                    metric_frame["ranked_probability_score"].mean()
                ),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["mean_log_loss", "mean_brier_score", "alpha_on_dixon_coles"],
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

    print(json.dumps({"best_configuration": best_row}, indent=2))
    print(f"Saved blend sweep to {summary_output}")


if __name__ == "__main__":
    main()
