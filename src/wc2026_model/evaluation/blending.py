from __future__ import annotations

import pandas as pd

from wc2026_model.evaluation.scoring import (
    brier_score_three_way,
    log_loss_three_way,
    ranked_probability_score,
)
from wc2026_model.models import blend_three_way_probabilities
from wc2026_model.types import ThreeWayProbabilities

_REQUIRED_PREDICTION_COLUMNS = {
    "model_name",
    "cutoff_date",
    "match_date",
    "home_team",
    "away_team",
    "actual_outcome",
    "pred_home",
    "pred_draw",
    "pred_away",
}


def build_convex_blend_predictions(
    predictions: pd.DataFrame,
    *,
    base_model_name: str,
    overlay_model_name: str,
    blended_model_name: str,
    alpha_on_base: float,
) -> pd.DataFrame:
    if not 0.0 <= alpha_on_base <= 1.0:
        raise ValueError(f"alpha_on_base must lie in [0, 1], got {alpha_on_base}.")

    missing_columns = _REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Predictions frame is missing required columns: {missing}")

    base_predictions = predictions.loc[predictions["model_name"] == base_model_name].copy()
    overlay_predictions = predictions.loc[predictions["model_name"] == overlay_model_name].copy()
    if base_predictions.empty:
        raise ValueError(f"No prediction rows found for base model {base_model_name!r}.")
    if overlay_predictions.empty:
        raise ValueError(f"No prediction rows found for overlay model {overlay_model_name!r}.")

    merge_keys = ["cutoff_date", "match_date", "home_team", "away_team", "actual_outcome"]
    merged = base_predictions.merge(
        overlay_predictions.loc[:, merge_keys + ["pred_home", "pred_draw", "pred_away"]],
        on=merge_keys,
        suffixes=("_base", "_overlay"),
        how="inner",
    )
    if merged.empty:
        raise ValueError(
            "Base and overlay predictions did not share any common match keys to blend."
        )

    blended_rows: list[dict[str, object]] = []
    passthrough_columns = [
        column
        for column in base_predictions.columns
        if column not in {"model_name", "pred_home", "pred_draw", "pred_away", "log_loss", "brier_score", "ranked_probability_score"}
    ]
    for row in merged.itertuples(index=False):
        base_probabilities = ThreeWayProbabilities(
            home=float(row.pred_home_base),
            draw=float(row.pred_draw_base),
            away=float(row.pred_away_base),
        )
        overlay_probabilities = ThreeWayProbabilities(
            home=float(row.pred_home_overlay),
            draw=float(row.pred_draw_overlay),
            away=float(row.pred_away_overlay),
        )
        blended_probabilities = blend_three_way_probabilities(
            base_probabilities,
            overlay_probabilities,
            alpha_on_base=alpha_on_base,
        )

        blended_row = {"model_name": blended_model_name}
        for column in passthrough_columns:
            blended_row[column] = getattr(row, column)
        blended_row["pred_home"] = blended_probabilities.home
        blended_row["pred_draw"] = blended_probabilities.draw
        blended_row["pred_away"] = blended_probabilities.away
        blended_row["log_loss"] = log_loss_three_way(
            blended_probabilities,
            str(row.actual_outcome),
        )
        blended_row["brier_score"] = brier_score_three_way(
            blended_probabilities,
            str(row.actual_outcome),
        )
        blended_row["ranked_probability_score"] = ranked_probability_score(
            blended_probabilities,
            str(row.actual_outcome),
        )
        blended_rows.append(blended_row)

    return pd.DataFrame.from_records(blended_rows)
