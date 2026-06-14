import pandas as pd
import pytest

from wc2026_model.evaluation.blending import build_convex_blend_predictions


def test_build_convex_blend_predictions_interpolates_probabilities() -> None:
    predictions = pd.DataFrame(
        [
            {
                "model_name": "base",
                "cutoff_date": "2025-01-01",
                "match_date": "2025-01-15",
                "home_team": "France",
                "away_team": "Senegal",
                "actual_outcome": "home",
                "pred_home": 0.50,
                "pred_draw": 0.30,
                "pred_away": 0.20,
                "elo_diff_pre": 120.0,
            },
            {
                "model_name": "overlay",
                "cutoff_date": "2025-01-01",
                "match_date": "2025-01-15",
                "home_team": "France",
                "away_team": "Senegal",
                "actual_outcome": "home",
                "pred_home": 0.60,
                "pred_draw": 0.20,
                "pred_away": 0.20,
                "elo_diff_pre": 120.0,
            },
        ]
    )

    blended = build_convex_blend_predictions(
        predictions,
        base_model_name="base",
        overlay_model_name="overlay",
        blended_model_name="blend",
        alpha_on_base=0.75,
    )

    assert len(blended) == 1
    assert blended.loc[0, "model_name"] == "blend"
    assert blended.loc[0, "pred_home"] == pytest.approx(0.525)
    assert blended.loc[0, "pred_draw"] == pytest.approx(0.275)
    assert blended.loc[0, "pred_away"] == pytest.approx(0.20)
