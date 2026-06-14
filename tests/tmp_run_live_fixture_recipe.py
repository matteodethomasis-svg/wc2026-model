from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pandas as pd


def test_run_live_fixture_recipe() -> None:
    previous_argv = sys.argv[:]
    try:
        sys.argv = [
            "predict_world_cup_fixtures.py",
            "--model-input",
            "models/baseline_dixon_coles_elo.pkl",
            "--fixtures-input",
            "data/raw/international_results.csv",
            "--elo-ratings-input",
            "reports/baseline_latest_elo_ratings.csv",
            "--training-frame-input",
            "reports/baseline_training_frame.csv",
            "--elo-blend-alpha",
            "0.75",
            "--calibration-gamma-home",
            "1.05",
            "--calibration-gamma-draw",
            "1.0",
            "--calibration-gamma-away",
            "1.1",
            "--squad-strength-input",
            "reports/wc2026_squad_strength_ratings.csv",
            "--squad-strength-column",
            "squad_club_elo_rating",
            "--squad-elo-scale",
            "0.75",
            "--start-date",
            "2026-06-13",
            "--output",
            "reports/wc2026_fixture_predictions_recipe_2026-06-13.csv",
            "--summary-output",
            "reports/wc2026_fixture_predictions_recipe_2026-06-13_summary.json",
        ]
        runpy.run_path("scripts/predict_world_cup_fixtures.py", run_name="__main__")
    finally:
        sys.argv = previous_argv

    summary = json.loads(
        Path("reports/wc2026_fixture_predictions_recipe_2026-06-13_summary.json").read_text(
            encoding="utf-8"
        )
    )
    predictions = pd.read_csv("reports/wc2026_fixture_predictions_recipe_2026-06-13.csv")
    print(json.dumps(summary, indent=2))
    print(
        predictions.head(20).loc[
            :,
            [
                "match_date",
                "home_team",
                "away_team",
                "adjusted_elo_diff_pre",
                "home_win_probability",
                "draw_probability",
                "away_win_probability",
            ],
        ].to_string(index=False)
    )
