from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from wc2026_model.evaluation.scoring import (
    brier_score_three_way,
    log_loss_three_way,
    ranked_probability_score,
)
from wc2026_model.features import augment_with_pre_match_elo
from wc2026_model.pipeline import BaselineTrainingConfig, train_baseline_model
from wc2026_model.types import ThreeWayProbabilities


@dataclass(frozen=True)
class BacktestFoldSummary:
    cutoff_date: pd.Timestamp
    test_window_days: int
    train_matches: int
    test_matches: int
    average_log_loss: float
    average_brier_score: float
    average_ranked_probability_score: float


def generate_rolling_cutoffs(
    results: pd.DataFrame,
    *,
    start_date: str,
    end_date: str | None = None,
    step_days: int = 90,
) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date)
    finish = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(results["match_date"].max())
    cutoffs: list[pd.Timestamp] = []
    current = start
    while current <= finish:
        cutoffs.append(current)
        current += timedelta(days=step_days)
    return cutoffs


def run_expanding_window_backtest(
    results: pd.DataFrame,
    *,
    config: BaselineTrainingConfig,
    cutoffs: list[pd.Timestamp],
    test_window_days: int = 90,
) -> tuple[pd.DataFrame, list[BacktestFoldSummary]]:
    standardized = results.copy()
    standardized["match_date"] = pd.to_datetime(standardized["match_date"], errors="raise")
    standardized = standardized.sort_values(
        ["match_date", "home_team", "away_team"], kind="stable"
    ).reset_index(drop=True)
    full_with_elo = augment_with_pre_match_elo(standardized, config=config.elo_config)

    prediction_rows: list[dict[str, object]] = []
    fold_summaries: list[BacktestFoldSummary] = []

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
        test_frame = test_frame[
            test_frame["home_team"].isin(model.teams) & test_frame["away_team"].isin(model.teams)
        ].reset_index(drop=True)
        if test_frame.empty or training_frame.empty:
            continue

        fold_metrics = []
        for row in test_frame.itertuples(index=False):
            probabilities = model.predict_outcome_probabilities(
                row.home_team,
                row.away_team,
                neutral_site=bool(row.neutral),
                elo_diff_pre=float(row.elo_diff_pre),
                max_goals=10,
            )
            metrics = _score_prediction(probabilities, str(row.home_result))
            fold_metrics.append(metrics)
            prediction_rows.append(
                {
                    "cutoff_date": cutoff,
                    "match_date": row.match_date,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "actual_outcome": row.home_result,
                    "pred_home": probabilities.home,
                    "pred_draw": probabilities.draw,
                    "pred_away": probabilities.away,
                    "elo_diff_pre": row.elo_diff_pre,
                    **metrics,
                }
            )

        fold_metric_frame = pd.DataFrame(fold_metrics)
        fold_summaries.append(
            BacktestFoldSummary(
                cutoff_date=cutoff,
                test_window_days=test_window_days,
                train_matches=int(len(training_frame)),
                test_matches=int(len(test_frame)),
                average_log_loss=float(fold_metric_frame["log_loss"].mean()),
                average_brier_score=float(fold_metric_frame["brier_score"].mean()),
                average_ranked_probability_score=float(
                    fold_metric_frame["ranked_probability_score"].mean()
                ),
            )
        )

    return pd.DataFrame(prediction_rows), fold_summaries


def summarize_backtest_folds(fold_summaries: list[BacktestFoldSummary]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cutoff_date": summary.cutoff_date,
                "test_window_days": summary.test_window_days,
                "train_matches": summary.train_matches,
                "test_matches": summary.test_matches,
                "average_log_loss": summary.average_log_loss,
                "average_brier_score": summary.average_brier_score,
                "average_ranked_probability_score": summary.average_ranked_probability_score,
            }
            for summary in fold_summaries
        ]
    )


def _score_prediction(
    probabilities: ThreeWayProbabilities,
    actual_outcome: str,
) -> dict[str, float]:
    return {
        "log_loss": log_loss_three_way(probabilities, actual_outcome),
        "brier_score": brier_score_three_way(probabilities, actual_outcome),
        "ranked_probability_score": ranked_probability_score(probabilities, actual_outcome),
    }
