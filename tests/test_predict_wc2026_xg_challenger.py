import runpy

import pandas as pd

from wc2026_model.data import load_international_results
from wc2026_model.features import WorldCupXGConfig
from wc2026_model.pipeline import BaselineTrainingConfig


_SCRIPT_GLOBALS = runpy.run_path("scripts/predict_wc2026_xg_challenger.py")


def test_build_wc2026_xg_challenger_outputs_produces_fixture_rows(tmp_path) -> None:
    statsbomb_results = load_international_results("data/interim/statsbomb_world_cup_match_features.csv")
    fixture_base = pd.DataFrame(
        [
            {
                "match_id": "2026-06-12::Brazil::Morocco",
                "match_date": "2026-06-12",
                "home_team": "Brazil",
                "away_team": "Morocco",
                "neutral": True,
                "adjusted_elo_diff_pre": 12.0,
                "home_win_probability": 0.45,
                "draw_probability": 0.28,
                "away_win_probability": 0.27,
            },
            {
                "match_id": "2026-06-13::France::Switzerland",
                "match_date": "2026-06-13",
                "home_team": "France",
                "away_team": "Switzerland",
                "neutral": True,
                "adjusted_elo_diff_pre": 55.0,
                "home_win_probability": 0.57,
                "draw_probability": 0.24,
                "away_win_probability": 0.19,
            },
        ]
    )

    predictions, comparison, summary = _SCRIPT_GLOBALS["build_wc2026_xg_challenger_outputs"](
        statsbomb_results=statsbomb_results,
        fixture_base=fixture_base,
        training_config=BaselineTrainingConfig(
            min_match_date="2018-01-01",
            min_team_matches=1,
        ),
        xg_config=WorldCupXGConfig(window_size=3),
        fixture_elo_column="adjusted_elo_diff_pre",
    )

    assert len(predictions) == 2
    assert len(comparison) == 2
    assert summary["fixture_count"] == 2
    assert "max_absolute_probability_delta" in comparison.columns
    assert predictions["home_win_probability"].between(0.0, 1.0).all()
