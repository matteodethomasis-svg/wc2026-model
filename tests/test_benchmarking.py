import pandas as pd
import pytest

from wc2026_model.evaluation import (
    EloMultinomialBenchmark,
    FormEloMultinomialBenchmark,
    XGEloMultinomialBenchmark,
    expected_calibration_error_three_way,
    uniform_three_way_probabilities,
    weighted_outcome_frequencies,
)
from wc2026_model.features import (
    augment_with_pre_match_elo,
    augment_with_pre_match_form_features,
    augment_with_pre_match_xg_features,
)
from wc2026_model.types import ThreeWayProbabilities


def test_weighted_outcome_frequencies_returns_valid_probabilities(
    sample_international_results: pd.DataFrame,
) -> None:
    frame = augment_with_pre_match_elo(sample_international_results).copy()
    frame["sample_weight"] = 1.0
    probabilities = weighted_outcome_frequencies(frame)

    total = probabilities.home + probabilities.draw + probabilities.away
    assert total == pytest.approx(1.0)
    assert probabilities.home > 0.0


def test_elo_multinomial_benchmark_produces_valid_probabilities(
    sample_international_results: pd.DataFrame,
) -> None:
    frame = augment_with_pre_match_elo(sample_international_results).copy()
    frame["sample_weight"] = 1.0
    benchmark = EloMultinomialBenchmark.fit(frame)

    probabilities = benchmark.predict_proba(frame.iloc[0])
    total = probabilities.home + probabilities.draw + probabilities.away
    assert total == pytest.approx(1.0, abs=1e-8)
    assert min(probabilities.home, probabilities.draw, probabilities.away) >= 0.0


def test_form_elo_multinomial_benchmark_produces_valid_probabilities(
    sample_international_results: pd.DataFrame,
) -> None:
    frame = augment_with_pre_match_elo(sample_international_results).copy()
    frame["sample_weight"] = 1.0
    frame = augment_with_pre_match_form_features(frame)
    benchmark = FormEloMultinomialBenchmark.fit(frame)

    probabilities = benchmark.predict_proba(frame.iloc[-1])
    total = probabilities.home + probabilities.draw + probabilities.away
    assert total == pytest.approx(1.0, abs=1e-8)
    assert min(probabilities.home, probabilities.draw, probabilities.away) >= 0.0


def test_xg_elo_multinomial_benchmark_produces_valid_probabilities() -> None:
    frame = pd.DataFrame(
        [
            {
                "match_date": "2024-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_goals": 2,
                "away_goals": 1,
                "neutral": True,
                "home_result": "home",
                "elo_diff_pre": 40.0,
                "home_xg": 1.8,
                "away_xg": 0.9,
                "home_shots": 12.0,
                "away_shots": 8.0,
                "home_shots_on_target": 5.0,
                "away_shots_on_target": 3.0,
            },
            {
                "match_date": "2024-01-07",
                "home_team": "Gamma",
                "away_team": "Alpha",
                "home_goals": 0,
                "away_goals": 1,
                "neutral": True,
                "home_result": "away",
                "elo_diff_pre": -35.0,
                "home_xg": 0.5,
                "away_xg": 1.4,
                "home_shots": 7.0,
                "away_shots": 11.0,
                "home_shots_on_target": 2.0,
                "away_shots_on_target": 4.0,
            },
            {
                "match_date": "2024-01-14",
                "home_team": "Beta",
                "away_team": "Gamma",
                "home_goals": 1,
                "away_goals": 1,
                "neutral": True,
                "home_result": "draw",
                "elo_diff_pre": 10.0,
                "home_xg": 1.2,
                "away_xg": 1.1,
                "home_shots": 10.0,
                "away_shots": 9.0,
                "home_shots_on_target": 4.0,
                "away_shots_on_target": 3.0,
            },
            {
                "match_date": "2024-01-21",
                "home_team": "Alpha",
                "away_team": "Gamma",
                "home_goals": 3,
                "away_goals": 1,
                "neutral": True,
                "home_result": "home",
                "elo_diff_pre": 55.0,
                "home_xg": 2.0,
                "away_xg": 0.8,
                "home_shots": 15.0,
                "away_shots": 6.0,
                "home_shots_on_target": 6.0,
                "away_shots_on_target": 2.0,
            },
        ]
    )
    frame["sample_weight"] = 1.0
    frame = augment_with_pre_match_xg_features(frame)
    benchmark = XGEloMultinomialBenchmark.fit(frame)

    probabilities = benchmark.predict_proba(frame.iloc[-1])
    total = probabilities.home + probabilities.draw + probabilities.away
    assert total == pytest.approx(1.0, abs=1e-8)
    assert min(probabilities.home, probabilities.draw, probabilities.away) >= 0.0


def test_xg_elo_multinomial_benchmark_context_variant_produces_valid_probabilities() -> None:
    frame = pd.DataFrame(
        [
            {
                "match_date": "2024-01-01",
                "home_team": "France",
                "away_team": "Brazil",
                "home_goals": 2,
                "away_goals": 1,
                "neutral": True,
                "home_result": "home",
                "elo_diff_pre": 40.0,
                "home_xg": 1.8,
                "away_xg": 0.9,
                "home_shots": 12.0,
                "away_shots": 8.0,
                "home_shots_on_target": 5.0,
                "away_shots_on_target": 3.0,
            },
            {
                "match_date": "2024-01-07",
                "home_team": "Brazil",
                "away_team": "France",
                "home_goals": 1,
                "away_goals": 1,
                "neutral": True,
                "home_result": "draw",
                "elo_diff_pre": -10.0,
                "home_xg": 1.3,
                "away_xg": 1.1,
                "home_shots": 11.0,
                "away_shots": 9.0,
                "home_shots_on_target": 4.0,
                "away_shots_on_target": 4.0,
            },
            {
                "match_date": "2024-01-14",
                "home_team": "Spain",
                "away_team": "Japan",
                "home_goals": 1,
                "away_goals": 0,
                "neutral": True,
                "home_result": "home",
                "elo_diff_pre": 55.0,
                "home_xg": 1.6,
                "away_xg": 0.7,
                "home_shots": 14.0,
                "away_shots": 6.0,
                "home_shots_on_target": 6.0,
                "away_shots_on_target": 2.0,
            },
            {
                "match_date": "2024-01-21",
                "home_team": "France",
                "away_team": "Brazil",
                "home_goals": 0,
                "away_goals": 1,
                "neutral": True,
                "home_result": "away",
                "elo_diff_pre": 15.0,
                "home_xg": 0.8,
                "away_xg": 1.4,
                "home_shots": 9.0,
                "away_shots": 12.0,
                "home_shots_on_target": 2.0,
                "away_shots_on_target": 5.0,
            },
        ]
    )
    frame["sample_weight"] = 1.0
    frame = augment_with_pre_match_xg_features(frame)
    benchmark = XGEloMultinomialBenchmark.fit(frame, include_context_features=True)

    probabilities = benchmark.predict_proba(frame.iloc[-1])
    total = probabilities.home + probabilities.draw + probabilities.away
    assert total == pytest.approx(1.0, abs=1e-8)
    assert min(probabilities.home, probabilities.draw, probabilities.away) >= 0.0
    assert benchmark.context_feature_groups == ("shot_accuracy", "confederation", "h2h")
    assert "same_confederation" in benchmark.feature_columns
    assert "h2h_match_count_scaled" in benchmark.feature_columns


def test_xg_elo_multinomial_benchmark_supports_context_ablation_groups() -> None:
    frame = pd.DataFrame(
        [
            {
                "match_date": "2024-01-01",
                "home_team": "France",
                "away_team": "Brazil",
                "home_goals": 2,
                "away_goals": 1,
                "neutral": True,
                "home_result": "home",
                "elo_diff_pre": 40.0,
                "home_xg": 1.8,
                "away_xg": 0.9,
                "home_shots": 12.0,
                "away_shots": 8.0,
                "home_shots_on_target": 5.0,
                "away_shots_on_target": 3.0,
            },
            {
                "match_date": "2024-01-07",
                "home_team": "Brazil",
                "away_team": "France",
                "home_goals": 1,
                "away_goals": 1,
                "neutral": True,
                "home_result": "draw",
                "elo_diff_pre": -10.0,
                "home_xg": 1.3,
                "away_xg": 1.1,
                "home_shots": 11.0,
                "away_shots": 9.0,
                "home_shots_on_target": 4.0,
                "away_shots_on_target": 4.0,
            },
            {
                "match_date": "2024-01-14",
                "home_team": "Spain",
                "away_team": "Japan",
                "home_goals": 1,
                "away_goals": 0,
                "neutral": True,
                "home_result": "home",
                "elo_diff_pre": 55.0,
                "home_xg": 1.6,
                "away_xg": 0.7,
                "home_shots": 14.0,
                "away_shots": 6.0,
                "home_shots_on_target": 6.0,
                "away_shots_on_target": 2.0,
            },
            {
                "match_date": "2024-01-21",
                "home_team": "France",
                "away_team": "Brazil",
                "home_goals": 0,
                "away_goals": 1,
                "neutral": True,
                "home_result": "away",
                "elo_diff_pre": 15.0,
                "home_xg": 0.8,
                "away_xg": 1.4,
                "home_shots": 9.0,
                "away_shots": 12.0,
                "home_shots_on_target": 2.0,
                "away_shots_on_target": 5.0,
            },
        ]
    )
    frame["sample_weight"] = 1.0
    frame = augment_with_pre_match_xg_features(frame)

    shot_accuracy = XGEloMultinomialBenchmark.fit(
        frame,
        context_feature_groups=("shot_accuracy",),
    )
    confederation = XGEloMultinomialBenchmark.fit(
        frame,
        context_feature_groups=("confederation",),
    )
    h2h = XGEloMultinomialBenchmark.fit(frame, context_feature_groups=("h2h",))
    h2h_decayed = XGEloMultinomialBenchmark.fit(
        frame,
        context_feature_groups=("h2h_decayed",),
    )

    assert "shot_accuracy_for_diff" in shot_accuracy.feature_columns
    assert "same_confederation" not in shot_accuracy.feature_columns
    assert "h2h_match_count_scaled" not in shot_accuracy.feature_columns
    assert "h2h_decayed_weight_scaled" not in shot_accuracy.feature_columns
    assert "same_confederation" in confederation.feature_columns
    assert "shot_accuracy_for_diff" not in confederation.feature_columns
    assert "h2h_match_count_scaled" not in confederation.feature_columns
    assert "h2h_match_count_scaled" in h2h.feature_columns
    assert "same_confederation" not in h2h.feature_columns
    assert "shot_accuracy_for_diff" not in h2h.feature_columns
    # Raw and decayed h2h are distinct families: selecting one must not pull the other.
    assert "h2h_decayed_weight_scaled" not in h2h.feature_columns
    assert "h2h_decayed_weight_scaled" in h2h_decayed.feature_columns
    assert "h2h_decayed_home_win_rate" in h2h_decayed.feature_columns
    assert "h2h_match_count_scaled" not in h2h_decayed.feature_columns
    assert "same_confederation" not in h2h_decayed.feature_columns


def test_augment_with_pre_match_form_features_is_pre_match() -> None:
    frame = pd.DataFrame(
        [
            {
                "match_date": "2024-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_goals": 2,
                "away_goals": 1,
            },
            {
                "match_date": "2024-01-10",
                "home_team": "Alpha",
                "away_team": "Gamma",
                "home_goals": 1,
                "away_goals": 1,
            },
            {
                "match_date": "2024-01-20",
                "home_team": "Beta",
                "away_team": "Gamma",
                "home_goals": 0,
                "away_goals": 3,
            },
        ]
    )

    augmented = augment_with_pre_match_form_features(frame)

    assert augmented.loc[0, "home_form_match_count"] == pytest.approx(0.0)
    assert pd.isna(augmented.loc[0, "home_form_points_per_match"])
    assert augmented.loc[1, "home_form_match_count"] == pytest.approx(1.0)
    assert augmented.loc[1, "home_form_points_per_match"] == pytest.approx(3.0)
    assert augmented.loc[1, "home_form_goal_diff_per_match"] == pytest.approx(1.0)
    assert augmented.loc[1, "home_days_since_last_match"] == pytest.approx(9.0)
    assert augmented.loc[2, "home_form_points_per_match"] == pytest.approx(0.0)
    assert augmented.loc[2, "away_form_points_per_match"] == pytest.approx(1.0)


def test_expected_calibration_error_three_way_is_bounded() -> None:
    frame = pd.DataFrame(
        [
            {
                "actual_outcome": "home",
                "pred_home": 0.70,
                "pred_draw": 0.20,
                "pred_away": 0.10,
            },
            {
                "actual_outcome": "away",
                "pred_home": 0.10,
                "pred_draw": 0.20,
                "pred_away": 0.70,
            },
            {
                "actual_outcome": "draw",
                "pred_home": 0.20,
                "pred_draw": 0.60,
                "pred_away": 0.20,
            },
        ]
    )

    ece = expected_calibration_error_three_way(frame, bins=5)
    assert 0.0 <= ece <= 1.0


def test_uniform_three_way_probabilities_is_flat() -> None:
    probabilities = uniform_three_way_probabilities()
    assert probabilities == ThreeWayProbabilities(
        home=pytest.approx(1.0 / 3.0),
        draw=pytest.approx(1.0 / 3.0),
        away=pytest.approx(1.0 / 3.0),
    )
