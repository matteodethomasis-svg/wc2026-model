from __future__ import annotations

import runpy

import pandas as pd
import pytest


_SCRIPT_GLOBALS = runpy.run_path("scripts/predict_world_cup_fixtures.py")


def test_availability_elo_adjustment_penalizes_missing_starters_and_goalkeeper() -> None:
    adjustment = _SCRIPT_GLOBALS["_availability_elo_adjustment"](
        {
            "expected_starter_count": 11,
            "expected_starter_availability_weight_sum": 9.5,
            "expected_goalkeeper_count": 1,
            "goalkeeper_starter_available": False,
            "lineup_confidence": 0.8,
        },
        starter_absence_elo=18.0,
        goalkeeper_absence_elo=24.0,
    )

    assert adjustment == pytest.approx(-40.8)


def test_availability_elo_adjustment_ignores_missing_feed() -> None:
    adjustment = _SCRIPT_GLOBALS["_availability_elo_adjustment"](
        {
            "expected_starter_count": 0,
            "expected_starter_availability_weight_sum": 0.0,
            "expected_goalkeeper_count": 0,
            "goalkeeper_starter_available": False,
            "lineup_confidence": 0.0,
        },
        starter_absence_elo=18.0,
        goalkeeper_absence_elo=24.0,
    )

    assert adjustment == 0.0


def test_build_and_get_team_availability_lookup_prefers_match_date_key() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "match_date": "2026-06-16",
                "team": "France",
                "expected_starter_count": 11,
                "expected_starter_availability_weight_sum": 10.5,
                "expected_goalkeeper_count": 1,
                "goalkeeper_starter_available": True,
                "lineup_confidence": 0.9,
            },
            {
                "team": "France",
                "expected_starter_count": 9,
                "expected_starter_availability_weight_sum": 8.5,
                "expected_goalkeeper_count": 1,
                "goalkeeper_starter_available": False,
                "lineup_confidence": 0.6,
            },
        ]
    )

    lookup = _SCRIPT_GLOBALS["_load_team_availability_lookup_from_frame"](dataframe)
    dated = _SCRIPT_GLOBALS["_get_team_availability_record"](
        lookup,
        match_date=pd.Timestamp("2026-06-16"),
        team="France",
    )
    fallback = _SCRIPT_GLOBALS["_get_team_availability_record"](
        lookup,
        match_date=pd.Timestamp("2026-06-20"),
        team="France",
    )

    assert dated["lineup_confidence"] == 0.9
    assert fallback["lineup_confidence"] == 0.6
