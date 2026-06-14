"""Evaluation metrics for football forecasts."""

from .benchmarking import (
    EloMultinomialBenchmark,
    FormEloMultinomialBenchmark,
    XG_CONTEXT_FEATURE_GROUPS,
    XG_CONTEXT_LEGACY_BUNDLE,
    XGEloMultinomialBenchmark,
    uniform_three_way_probabilities,
    weighted_outcome_frequencies,
)
from .backtest import (
    BacktestFoldSummary,
    generate_rolling_cutoffs,
    run_expanding_window_backtest,
    summarize_backtest_folds,
)
from .blending import build_convex_blend_predictions
from .prediction_ledger import (
    append_match_snapshot,
    append_outright_snapshot,
    score_match_ledger,
    summarize_track_record,
)
from .calibration import (
    expected_calibration_error_three_way,
    power_calibrate_prediction_frame,
    power_calibrate_three_way,
    probabilities_to_row,
)
from .scoring import (
    brier_score_three_way,
    log_loss_three_way,
    ranked_probability_score,
)

__all__ = [
    "BacktestFoldSummary",
    "EloMultinomialBenchmark",
    "FormEloMultinomialBenchmark",
    "XG_CONTEXT_FEATURE_GROUPS",
    "XG_CONTEXT_LEGACY_BUNDLE",
    "XGEloMultinomialBenchmark",
    "brier_score_three_way",
    "append_match_snapshot",
    "append_outright_snapshot",
    "build_convex_blend_predictions",
    "score_match_ledger",
    "summarize_track_record",
    "expected_calibration_error_three_way",
    "generate_rolling_cutoffs",
    "log_loss_three_way",
    "power_calibrate_prediction_frame",
    "power_calibrate_three_way",
    "probabilities_to_row",
    "ranked_probability_score",
    "run_expanding_window_backtest",
    "summarize_backtest_folds",
    "uniform_three_way_probabilities",
    "weighted_outcome_frequencies",
]
