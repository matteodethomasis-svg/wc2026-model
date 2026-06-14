import pandas as pd
import pytest

from wc2026_model.pipeline import BaselineTrainingConfig, train_baseline_model


def test_train_baseline_model_produces_valid_probabilities(
    sample_international_results: pd.DataFrame,
) -> None:
    model, training_frame = train_baseline_model(
        sample_international_results,
        config=BaselineTrainingConfig(
            min_match_date="2024-01-01",
            min_team_matches=1,
            time_decay_xi=0.0,
            l2_penalty=0.05,
            maxiter=200,
        ),
    )

    probabilities = model.predict_outcome_probabilities(
        "Alpha",
        "Gamma",
        neutral_site=True,
        elo_diff_pre=0.0,
        max_goals=8,
    )
    total_probability = probabilities.home + probabilities.draw + probabilities.away

    assert not training_frame.empty
    assert {"home_elo_pre", "away_elo_pre", "sample_weight"} <= set(training_frame.columns)
    assert model.fit_result.success is True
    assert model.fit_result.objective_value == pytest.approx(model.fit_result.objective_value)
    assert total_probability == pytest.approx(1.0, abs=1e-8)
    assert probabilities.home > probabilities.away
