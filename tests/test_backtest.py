import pandas as pd

from wc2026_model.evaluation import (
    generate_rolling_cutoffs,
    run_expanding_window_backtest,
    summarize_backtest_folds,
)
from wc2026_model.pipeline import BaselineTrainingConfig


def test_generate_rolling_cutoffs_returns_ordered_dates() -> None:
    frame = pd.DataFrame({"match_date": pd.to_datetime(["2024-01-01", "2024-12-31"])})
    cutoffs = generate_rolling_cutoffs(
        frame,
        start_date="2024-01-01",
        end_date="2024-07-01",
        step_days=90,
    )
    assert len(cutoffs) == 3
    assert cutoffs[0] < cutoffs[1] < cutoffs[2]


def test_run_expanding_window_backtest_produces_predictions(
    sample_international_results: pd.DataFrame,
) -> None:
    cutoffs = generate_rolling_cutoffs(
        sample_international_results,
        start_date="2024-02-01",
        end_date="2024-02-01",
        step_days=90,
    )
    predictions, fold_summaries = run_expanding_window_backtest(
        sample_international_results,
        config=BaselineTrainingConfig(
            min_match_date="2024-01-01",
            min_team_matches=1,
            time_decay_xi=0.0,
            l2_penalty=0.05,
            maxiter=200,
        ),
        cutoffs=cutoffs,
        test_window_days=60,
    )
    summary_frame = summarize_backtest_folds(fold_summaries)

    assert not predictions.empty
    assert not summary_frame.empty
    assert {"pred_home", "pred_draw", "pred_away"} <= set(predictions.columns)
    assert {"average_log_loss", "average_ranked_probability_score"} <= set(summary_frame.columns)
