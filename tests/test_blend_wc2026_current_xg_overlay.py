import runpy

import pandas as pd
import pytest


_SCRIPT_GLOBALS = runpy.run_path("scripts/blend_wc2026_current_xg_overlay.py")


def test_build_gated_xg_overlay_applies_coverage_and_divergence_gates() -> None:
    current = pd.DataFrame(
        [
            {
                "match_id": "1",
                "match_date": "2026-06-12",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_win_probability": 0.50,
                "draw_probability": 0.25,
                "away_win_probability": 0.25,
            },
            {
                "match_id": "2",
                "match_date": "2026-06-13",
                "home_team": "Gamma",
                "away_team": "Delta",
                "home_win_probability": 0.60,
                "draw_probability": 0.22,
                "away_win_probability": 0.18,
            },
        ]
    )
    xg = pd.DataFrame(
        [
            {
                "match_id": "1",
                "match_date": "2026-06-12",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_xg_match_count": 3.0,
                "away_xg_match_count": 3.0,
                "both_teams_have_xg_history": True,
                "home_win_probability": 0.20,
                "draw_probability": 0.50,
                "away_win_probability": 0.30,
            },
            {
                "match_id": "2",
                "match_date": "2026-06-13",
                "home_team": "Gamma",
                "away_team": "Delta",
                "home_xg_match_count": 0.0,
                "away_xg_match_count": 3.0,
                "both_teams_have_xg_history": False,
                "home_win_probability": 0.10,
                "draw_probability": 0.60,
                "away_win_probability": 0.30,
            },
        ]
    )

    blended, summary = _SCRIPT_GLOBALS["build_gated_xg_overlay"](
        current_predictions=current,
        xg_predictions=xg,
        min_xg_matches_per_team=3.0,
        max_xg_weight=0.35,
        delta_soft_cap=0.20,
    )

    first = blended.loc[blended["match_id"] == "1"].iloc[0]
    second = blended.loc[blended["match_id"] == "2"].iloc[0]

    assert first["xg_overlay_weight"] == pytest.approx(0.23333333333333334)
    assert first["home_win_probability"] < first["current_home_win_probability"]
    assert second["xg_overlay_weight"] == pytest.approx(0.0)
    assert second["home_win_probability"] == pytest.approx(second["current_home_win_probability"])
    assert summary["fixtures_with_positive_xg_weight"] == 1
