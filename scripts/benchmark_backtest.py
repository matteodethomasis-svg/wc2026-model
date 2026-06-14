from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from wc2026_model.data import download_international_results_csv, load_international_results
from wc2026_model.evaluation import (
    EloMultinomialBenchmark,
    FormEloMultinomialBenchmark,
    XG_CONTEXT_FEATURE_GROUPS,
    XGEloMultinomialBenchmark,
    expected_calibration_error_three_way,
    generate_rolling_cutoffs,
    probabilities_to_row,
    uniform_three_way_probabilities,
    weighted_outcome_frequencies,
)
from wc2026_model.evaluation.scoring import (
    brier_score_three_way,
    log_loss_three_way,
    ranked_probability_score,
)
from wc2026_model.features import (
    WorldCupXGConfig,
    attach_confederation_features,
    augment_with_pre_match_elo,
    augment_with_pre_match_form_features,
    augment_with_pre_match_h2h_features,
    augment_with_pre_match_xg_features,
)
from wc2026_model.models import blend_three_way_probabilities
from wc2026_model.pipeline import BaselineTrainingConfig, train_baseline_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the Dixon-Coles baseline against simpler models."
    )
    parser.add_argument("--input", default="data/interim/international_results_augmented.csv")
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument("--min-match-date", default="2010-01-01")
    parser.add_argument("--backtest-start", default="2024-01-01")
    parser.add_argument("--backtest-end", default="2026-06-11")
    parser.add_argument("--step-days", type=int, default=90)
    parser.add_argument("--test-window-days", type=int, default=90)
    parser.add_argument("--min-team-matches", type=int, default=10)
    parser.add_argument("--time-decay-xi", type=float, default=0.001)
    parser.add_argument("--l2-penalty", type=float, default=0.01)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument(
        "--xg-window-size",
        type=int,
        default=5,
        help="Rolling match window used when constructing pre-match xG features, if xG columns exist.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.75,
        help="Weight on Dixon-Coles inside the optional hybrid blend benchmark.",
    )
    parser.add_argument(
        "--disable-hybrid-blend",
        action="store_true",
        help="Skip the Dixon-Coles plus Elo hybrid benchmark variant.",
    )
    parser.add_argument(
        "--predictions-output",
        default="reports/benchmark_backtest_predictions.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/benchmark_backtest_summary.csv",
    )
    parser.add_argument(
        "--aggregate-output",
        default="reports/benchmark_backtest_aggregate.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
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

    predictions = run_benchmark_backtest(
        results=results,
        config=config,
        cutoffs=cutoffs,
        test_window_days=args.test_window_days,
        xg_config=WorldCupXGConfig(window_size=args.xg_window_size),
        hybrid_alpha=args.hybrid_alpha,
        include_hybrid_blend=not args.disable_hybrid_blend,
    )

    summary = summarize_benchmark_predictions(predictions)
    aggregate = build_aggregate_benchmark_report(summary)

    predictions_output = Path(args.predictions_output)
    summary_output = Path(args.summary_output)
    aggregate_output = Path(args.aggregate_output)
    for path in (predictions_output, summary_output, aggregate_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(predictions_output, index=False)
    summary.to_csv(summary_output, index=False)
    aggregate_output.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print(json.dumps(aggregate, indent=2))


def run_benchmark_backtest(
    *,
    results: pd.DataFrame,
    config: BaselineTrainingConfig,
    cutoffs: list[pd.Timestamp],
    test_window_days: int,
    xg_config: WorldCupXGConfig | None = None,
    hybrid_alpha: float = 0.75,
    include_hybrid_blend: bool = True,
) -> pd.DataFrame:
    xg_config = xg_config or WorldCupXGConfig()
    standardized = results.copy()
    standardized["match_date"] = pd.to_datetime(standardized["match_date"], errors="raise")
    standardized = standardized.sort_values(
        ["match_date", "home_team", "away_team"], kind="stable"
    ).reset_index(drop=True)
    full_with_elo = augment_with_pre_match_elo(standardized, config=config.elo_config)
    full_with_features = augment_with_pre_match_form_features(full_with_elo)
    full_with_xg = None
    if _has_xg_match_columns(full_with_elo):
        full_with_xg = augment_with_pre_match_xg_features(full_with_elo, config=xg_config)
        full_with_xg = attach_confederation_features(full_with_xg)
        full_with_xg = augment_with_pre_match_h2h_features(full_with_xg)

    prediction_rows: list[dict[str, object]] = []
    for cutoff in cutoffs:
        test_end = cutoff + timedelta(days=test_window_days)
        model, training_frame = train_baseline_model(
            standardized,
            config=BaselineTrainingConfig(
                min_match_date=config.min_match_date,
                training_cutoff=cutoff.strftime("%Y-%m-%d"),
                min_team_matches=config.min_team_matches,
                time_decay_xi=config.time_decay_xi,
                l2_penalty=config.l2_penalty,
                maxiter=config.maxiter,
                elo_config=config.elo_config,
            ),
        )
        test_frame = full_with_elo[
            (full_with_elo["match_date"] >= cutoff) & (full_with_elo["match_date"] < test_end)
        ].copy()
        test_frame = _attach_form_features(test_frame, full_with_features)
        if full_with_xg is not None:
            test_frame = _attach_xg_features(test_frame, full_with_xg)
        test_frame = test_frame[train_test_team_mask(test_frame, model.teams)].reset_index(drop=True)
        if test_frame.empty or training_frame.empty:
            continue

        prior_probabilities = weighted_outcome_frequencies(training_frame)
        elo_benchmark = EloMultinomialBenchmark.fit(training_frame)
        training_frame_with_form = _attach_form_features(training_frame, full_with_features)
        form_elo_benchmark = FormEloMultinomialBenchmark.fit(training_frame_with_form)
        training_frame_with_xg = (
            _attach_xg_features(training_frame, full_with_xg)
            if full_with_xg is not None
            else None
        )
        xg_elo_benchmark = (
            XGEloMultinomialBenchmark.fit(
                training_frame_with_xg,
                xg_config=xg_config,
                include_context_features=False,
            )
            if training_frame_with_xg is not None
            else None
        )
        xg_context_benchmarks = (
            {
                model_name: XGEloMultinomialBenchmark.fit(
                    training_frame_with_xg,
                    xg_config=xg_config,
                    context_feature_groups=context_groups,
                )
                for model_name, context_groups in _xg_context_benchmark_specs().items()
            }
            if training_frame_with_xg is not None
            else {}
        )
        benchmark_models = {
            "dixon_coles_elo": lambda row: model.predict_outcome_probabilities(
                row.home_team,
                row.away_team,
                neutral_site=bool(row.neutral),
                elo_diff_pre=float(row.elo_diff_pre),
                max_goals=10,
            ),
            "elo_multinomial": lambda row: elo_benchmark.predict_proba(row),
            "elo_multinomial_form": lambda row: form_elo_benchmark.predict_proba(row),
            "historical_prior": lambda row: prior_probabilities,
            "uniform": lambda row: uniform_three_way_probabilities(),
        }
        if xg_elo_benchmark is not None:
            benchmark_models["elo_multinomial_xg"] = lambda row: xg_elo_benchmark.predict_proba(row)
        for model_name, xg_context_benchmark in xg_context_benchmarks.items():
            benchmark_models[model_name] = (
                lambda row, benchmark=xg_context_benchmark: benchmark.predict_proba(row)
            )
        if include_hybrid_blend:
            benchmark_models["dixon_coles_elo_blend"] = lambda row: blend_three_way_probabilities(
                model.predict_outcome_probabilities(
                    row.home_team,
                    row.away_team,
                    neutral_site=bool(row.neutral),
                    elo_diff_pre=float(row.elo_diff_pre),
                    max_goals=10,
                ),
                elo_benchmark.predict_proba(row),
                alpha_on_base=hybrid_alpha,
            )
            benchmark_models["dixon_coles_elo_blend_form"] = (
                lambda row: blend_three_way_probabilities(
                    model.predict_outcome_probabilities(
                        row.home_team,
                        row.away_team,
                        neutral_site=bool(row.neutral),
                        elo_diff_pre=float(row.elo_diff_pre),
                        max_goals=10,
                    ),
                    form_elo_benchmark.predict_proba(row),
                    alpha_on_base=hybrid_alpha,
                )
            )
            if xg_elo_benchmark is not None:
                benchmark_models["dixon_coles_elo_blend_xg"] = (
                    lambda row: blend_three_way_probabilities(
                        model.predict_outcome_probabilities(
                            row.home_team,
                            row.away_team,
                            neutral_site=bool(row.neutral),
                            elo_diff_pre=float(row.elo_diff_pre),
                            max_goals=10,
                        ),
                        xg_elo_benchmark.predict_proba(row),
                        alpha_on_base=hybrid_alpha,
                    )
                )
            xg_context_elo_benchmark = xg_context_benchmarks.get("elo_multinomial_xg_context")
            if xg_context_elo_benchmark is not None:
                benchmark_models["dixon_coles_elo_blend_xg_context"] = (
                    lambda row: blend_three_way_probabilities(
                        model.predict_outcome_probabilities(
                            row.home_team,
                            row.away_team,
                            neutral_site=bool(row.neutral),
                            elo_diff_pre=float(row.elo_diff_pre),
                            max_goals=10,
                        ),
                        xg_context_elo_benchmark.predict_proba(row),
                        alpha_on_base=hybrid_alpha,
                    )
                )

        for row in test_frame.itertuples(index=False):
            for model_name, predictor in benchmark_models.items():
                probabilities = predictor(row)
                prediction_rows.append(
                    {
                        "model_name": model_name,
                        "cutoff_date": cutoff,
                        "match_date": row.match_date,
                        "home_team": row.home_team,
                        "away_team": row.away_team,
                        "actual_outcome": row.home_result,
                        "elo_diff_pre": row.elo_diff_pre,
                        **probabilities_to_row(probabilities),
                        "log_loss": log_loss_three_way(probabilities, str(row.home_result)),
                        "brier_score": brier_score_three_way(probabilities, str(row.home_result)),
                        "ranked_probability_score": ranked_probability_score(
                            probabilities, str(row.home_result)
                        ),
                    }
                )
    return pd.DataFrame(prediction_rows)


def summarize_benchmark_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "predictions",
                "mean_log_loss",
                "mean_brier_score",
                "mean_ranked_probability_score",
                "expected_calibration_error",
            ]
        )

    summary_rows = []
    for model_name, model_predictions in predictions.groupby("model_name", sort=True):
        summary_rows.append(
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
    return pd.DataFrame(summary_rows).sort_values("mean_log_loss", kind="stable").reset_index(
        drop=True
    )


def build_aggregate_benchmark_report(summary: pd.DataFrame) -> dict[str, object]:
    if summary.empty:
        return {"models": []}

    rows = summary.to_dict(orient="records")
    dixon_row = next((row for row in rows if row["model_name"] == "dixon_coles_elo"), None)
    if dixon_row is not None:
        for row in rows:
            row["log_loss_improvement_vs_dixon_coles"] = (
                float(row["mean_log_loss"]) - float(dixon_row["mean_log_loss"])
            )
    best_row = rows[0]
    return {
        "best_model_by_log_loss": best_row["model_name"],
        "models": rows,
    }


def train_test_team_mask(frame: pd.DataFrame, teams: list[str]) -> pd.Series:
    return frame["home_team"].isin(teams) & frame["away_team"].isin(teams)


def _xg_context_benchmark_specs() -> dict[str, tuple[str, ...]]:
    return {
        "elo_multinomial_xg_shot_accuracy": ("shot_accuracy",),
        "elo_multinomial_xg_confederation": ("confederation",),
        "elo_multinomial_xg_h2h": ("h2h",),
        "elo_multinomial_xg_h2h_decayed": ("h2h_decayed",),
        "elo_multinomial_xg_pass_completion": ("pass_completion",),
        "elo_multinomial_xg_pressures": ("pressures",),
        # Legacy bundle: pinned to raw groups so the historical 1.1366 baseline
        # stays reproducible. h2h_decayed is evaluated only as a standalone ablation.
        "elo_multinomial_xg_context": ("shot_accuracy", "confederation", "h2h"),
    }


def _attach_form_features(frame: pd.DataFrame, full_with_features: pd.DataFrame) -> pd.DataFrame:
    form_columns = [
        "home_form_match_count",
        "home_form_points_per_match",
        "home_form_goal_diff_per_match",
        "home_form_goals_for_per_match",
        "home_form_goals_against_per_match",
        "home_form_win_rate",
        "home_days_since_last_match",
        "away_form_match_count",
        "away_form_points_per_match",
        "away_form_goal_diff_per_match",
        "away_form_goals_for_per_match",
        "away_form_goals_against_per_match",
        "away_form_win_rate",
        "away_days_since_last_match",
    ]
    return frame.merge(
        full_with_features.loc[:, ["match_id"] + form_columns],
        on=["match_id"],
        how="left",
    )


def _attach_xg_features(frame: pd.DataFrame, full_with_xg_features: pd.DataFrame) -> pd.DataFrame:
    xg_columns = [
        "home_xg_match_count",
        "home_xg_for_per_match",
        "home_xg_against_per_match",
        "home_xg_diff_per_match",
        "home_shots_for_per_match",
        "home_shots_against_per_match",
        "home_shots_on_target_for_per_match",
        "home_shots_on_target_against_per_match",
        "home_shot_accuracy_for",
        "home_shot_accuracy_against",
        "home_xg_per_shot",
        "home_days_since_last_xg_match",
        "away_xg_match_count",
        "away_xg_for_per_match",
        "away_xg_against_per_match",
        "away_xg_diff_per_match",
        "away_shots_for_per_match",
        "away_shots_against_per_match",
        "away_shots_on_target_for_per_match",
        "away_shots_on_target_against_per_match",
        "away_shot_accuracy_for",
        "away_shot_accuracy_against",
        "away_xg_per_shot",
        "away_days_since_last_xg_match",
        "home_pass_completion_for",
        "home_pass_completion_against",
        "home_passes_for_per_match",
        "home_passes_against_per_match",
        "home_pressures_for_per_match",
        "home_pressures_against_per_match",
        "away_pass_completion_for",
        "away_pass_completion_against",
        "away_passes_for_per_match",
        "away_passes_against_per_match",
        "away_pressures_for_per_match",
        "away_pressures_against_per_match",
        "home_confederation",
        "away_confederation",
        "same_confederation",
        "h2h_match_count",
        "h2h_home_win_rate",
        "h2h_draw_rate",
        "h2h_away_win_rate",
        "h2h_decayed_match_weight",
        "h2h_decayed_home_win_rate",
        "h2h_decayed_draw_rate",
        "h2h_decayed_away_win_rate",
    ]
    for prefix in ("home", "away"):
        for confederation in ("afc", "caf", "concacaf", "conmebol", "ofc", "uefa"):
            xg_columns.append(f"{prefix}_is_{confederation}")
    return frame.merge(
        full_with_xg_features.loc[:, ["match_id"] + xg_columns],
        on=["match_id"],
        how="left",
    )


def _has_xg_match_columns(frame: pd.DataFrame) -> bool:
    return {"home_xg", "away_xg"}.issubset(frame.columns)


if __name__ == "__main__":
    main()
