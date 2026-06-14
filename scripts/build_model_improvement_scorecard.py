from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a compact scorecard showing which model upgrades improved validation metrics."
    )
    parser.add_argument(
        "--recipe-input",
        default="reports/current_best_model_recipe_2026-06-13.json",
    )
    parser.add_argument(
        "--squad-summary-input",
        default="reports/historical_world_cup_squad_scale_summary.csv",
    )
    parser.add_argument(
        "--expected-xi-summary-input",
        default="reports/historical_world_cup_expected_xi_scale_summary.csv",
    )
    parser.add_argument(
        "--mixed-summary-input",
        default="reports/historical_world_cup_expected_xi_plus_goalkeeper_scale_summary.csv",
    )
    parser.add_argument(
        "--json-output",
        default="reports/model_improvement_scorecard_2026-06-14.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="reports/model_improvement_scorecard_2026-06-14.md",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    recipe = json.loads(Path(args.recipe_input).read_text(encoding="utf-8"))
    squad_summary = pd.read_csv(args.squad_summary_input)
    expected_xi_summary = pd.read_csv(args.expected_xi_summary_input)
    mixed_summary = pd.read_csv(args.mixed_summary_input)

    generic = _build_generic_section(recipe)
    squad_layer = _build_single_scale_section(
        squad_summary,
        scale_column="scale",
        label="full_squad_strength",
    )
    expected_xi_layer = _build_single_scale_section(
        expected_xi_summary,
        scale_column="scale",
        label="expected_xi_strength",
    )
    mixed_layer = _build_mixed_layer_section(mixed_summary)

    scorecard = {
        "generic_backtest": generic,
        "world_cup_player_layers": {
            "full_squad_strength": squad_layer,
            "expected_xi_strength": expected_xi_layer,
            "expected_xi_plus_goalkeeper": mixed_layer,
        },
        "best_supported_recipe": {
            "generic": {
                "elo_blend_alpha": recipe["recommended_recipe"]["elo_blend_alpha"],
                "calibration_gamma_home": recipe["recommended_recipe"]["calibration_gamma_home"],
                "calibration_gamma_draw": recipe["recommended_recipe"]["calibration_gamma_draw"],
                "calibration_gamma_away": recipe["recommended_recipe"]["calibration_gamma_away"],
            },
            "player_layer": {
                "primary_rating_column": "expected_xi_club_elo_rating",
                "primary_scale": mixed_layer["best_configuration"]["primary_scale"],
                "secondary_rating_column": mixed_layer["best_configuration"]["secondary_rating_column"],
                "secondary_scale": mixed_layer["best_configuration"]["secondary_scale"],
            },
        },
    }

    json_output = Path(args.json_output)
    markdown_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    markdown_output.write_text(_render_markdown(scorecard), encoding="utf-8")

    print(json.dumps(scorecard, indent=2))
    print(f"Saved JSON scorecard to {json_output}")
    print(f"Saved Markdown scorecard to {markdown_output}")


def _build_generic_section(recipe: dict[str, object]) -> dict[str, object]:
    metrics = recipe["backtest_metrics"]
    baseline = metrics["dixon_coles_elo"]
    blend = metrics["dixon_coles_elo_blend_75_25"]
    calibrated = metrics["dixon_coles_elo_blend_75_25_calibrated"]
    return {
        "baseline": baseline,
        "blend": blend,
        "blend_plus_calibration": calibrated,
        "blend_vs_baseline": _improvement_block(
            baseline["mean_log_loss"],
            blend["mean_log_loss"],
        ),
        "blend_plus_calibration_vs_baseline": _improvement_block(
            baseline["mean_log_loss"],
            calibrated["mean_log_loss"],
        ),
    }


def _build_single_scale_section(
    summary: pd.DataFrame,
    *,
    scale_column: str,
    label: str,
) -> dict[str, object]:
    by_scale = _group_weighted_summary(summary, [scale_column])
    baseline_log_loss = float(by_scale.loc[by_scale[scale_column] == 0.0, "mean_log_loss"].iloc[0])
    best_row = by_scale.sort_values("mean_log_loss", kind="stable").iloc[0]
    best_log_loss = float(best_row["mean_log_loss"])
    return {
        "label": label,
        "baseline_no_player_adjustment_mean_log_loss": baseline_log_loss,
        "best_configuration": {
            scale_column: float(best_row[scale_column]),
            "mean_log_loss": float(best_log_loss),
        },
        "improvement_vs_no_player_adjustment": _improvement_block(
            baseline_log_loss,
            best_log_loss,
        ),
    }


def _build_mixed_layer_section(summary: pd.DataFrame) -> dict[str, object]:
    by_config = _group_weighted_summary(
        summary,
        ["primary_scale", "secondary_scale", "secondary_rating_column"],
    )
    baseline_log_loss = float(
        by_config.loc[
            (by_config["primary_scale"] == 0.0) & (by_config["secondary_scale"] == 0.0),
            "mean_log_loss",
        ].iloc[0]
    )
    best_row = by_config.sort_values("mean_log_loss", kind="stable").iloc[0]
    best_log_loss = float(best_row["mean_log_loss"])
    return {
        "baseline_no_player_adjustment_mean_log_loss": baseline_log_loss,
        "best_configuration": {
            "primary_scale": float(best_row["primary_scale"]),
            "secondary_scale": float(best_row["secondary_scale"]),
            "secondary_rating_column": str(best_row["secondary_rating_column"]),
            "mean_log_loss": float(best_log_loss),
        },
        "improvement_vs_no_player_adjustment": _improvement_block(
            baseline_log_loss,
            best_log_loss,
        ),
    }


def _group_weighted_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_columns, key, strict=True)}
        row["matches"] = int(pd.to_numeric(group["matches"], errors="raise").sum())
        row["mean_log_loss"] = _weighted_mean(group, "mean_log_loss")
        row["mean_brier_score"] = _weighted_mean(group, "mean_brier_score")
        row["mean_ranked_probability_score"] = _weighted_mean(
            group,
            "mean_ranked_probability_score",
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _weighted_mean(frame: pd.DataFrame, metric_column: str) -> float:
    if frame.empty:
        raise ValueError(f"Cannot compute weighted mean for empty frame on '{metric_column}'.")
    weights = pd.to_numeric(frame["matches"], errors="raise").astype(float)
    values = pd.to_numeric(frame[metric_column], errors="raise").astype(float)
    return float((values * weights).sum() / weights.sum())


def _improvement_block(baseline_value: float, improved_value: float) -> dict[str, float]:
    absolute = float(baseline_value - improved_value)
    relative_pct = float((absolute / baseline_value) * 100.0) if baseline_value else 0.0
    return {
        "baseline_mean_log_loss": float(baseline_value),
        "improved_mean_log_loss": float(improved_value),
        "absolute_log_loss_reduction": absolute,
        "relative_log_loss_reduction_pct": relative_pct,
    }


def _render_markdown(scorecard: dict[str, object]) -> str:
    generic = scorecard["generic_backtest"]
    full_squad = scorecard["world_cup_player_layers"]["full_squad_strength"]
    expected_xi = scorecard["world_cup_player_layers"]["expected_xi_strength"]
    mixed = scorecard["world_cup_player_layers"]["expected_xi_plus_goalkeeper"]

    return "\n".join(
        [
            "# Model Improvement Scorecard",
            "",
            "## Generic Backtest",
            "",
            (
                f"- Baseline log loss: `{generic['baseline']['mean_log_loss']:.6f}`\n"
                f"- Blend log loss: `{generic['blend']['mean_log_loss']:.6f}`\n"
                f"- Blend + calibration log loss: `{generic['blend_plus_calibration']['mean_log_loss']:.6f}`\n"
                f"- Blend improvement vs baseline: `{generic['blend_vs_baseline']['absolute_log_loss_reduction']:.6f}` "
                f"({generic['blend_vs_baseline']['relative_log_loss_reduction_pct']:.3f}%)\n"
                f"- Blend + calibration improvement vs baseline: "
                f"`{generic['blend_plus_calibration_vs_baseline']['absolute_log_loss_reduction']:.6f}` "
                f"({generic['blend_plus_calibration_vs_baseline']['relative_log_loss_reduction_pct']:.3f}%)"
            ),
            "",
            "## World Cup Player Layers",
            "",
            (
                f"- Full squad-strength best log loss: `{full_squad['best_configuration']['mean_log_loss']:.6f}` "
                f"at scale `{full_squad['best_configuration']['scale']:.2f}`\n"
                f"- Full squad-strength improvement vs no player adjustment: "
                f"`{full_squad['improvement_vs_no_player_adjustment']['absolute_log_loss_reduction']:.6f}` "
                f"({full_squad['improvement_vs_no_player_adjustment']['relative_log_loss_reduction_pct']:.3f}%)"
            ),
            "",
            (
                f"- Expected-XI best log loss: `{expected_xi['best_configuration']['mean_log_loss']:.6f}` "
                f"at scale `{expected_xi['best_configuration']['scale']:.2f}`\n"
                f"- Expected-XI improvement vs no player adjustment: "
                f"`{expected_xi['improvement_vs_no_player_adjustment']['absolute_log_loss_reduction']:.6f}` "
                f"({expected_xi['improvement_vs_no_player_adjustment']['relative_log_loss_reduction_pct']:.3f}%)"
            ),
            "",
            (
                f"- Expected-XI + goalkeeper best log loss: `{mixed['best_configuration']['mean_log_loss']:.6f}` "
                f"at primary scale `{mixed['best_configuration']['primary_scale']:.2f}` and goalkeeper scale "
                f"`{mixed['best_configuration']['secondary_scale']:.2f}`\n"
                f"- Mixed improvement vs no player adjustment: "
                f"`{mixed['improvement_vs_no_player_adjustment']['absolute_log_loss_reduction']:.6f}` "
                f"({mixed['improvement_vs_no_player_adjustment']['relative_log_loss_reduction_pct']:.3f}%)"
            ),
        ]
    )


if __name__ == "__main__":
    main()
