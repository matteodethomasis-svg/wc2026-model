from __future__ import annotations

import runpy

import pandas as pd


_SCRIPT_GLOBALS = runpy.run_path("scripts/evaluate_historical_world_cup_squad_scale.py")


def test_adjust_elo_diff_for_squad_strength_adds_scaled_team_gap() -> None:
    adjusted = _SCRIPT_GLOBALS["_adjust_elo_diff_for_squad_strength"](
        base_elo_diff=100.0,
        home_team="France",
        away_team="Norway",
        team_strength_lookup={"France": 1960.0, "Norway": 1820.0},
        scale=0.5,
    )

    assert adjusted == 170.0


def test_build_team_strength_lookup_filters_by_tournament_year() -> None:
    squad_strengths = pd.DataFrame(
        [
            {"tournament_year": 2018, "team": "France", "squad_club_elo_rating": 1900.0},
            {"tournament_year": 2022, "team": "France", "squad_club_elo_rating": 1950.0},
            {"tournament_year": 2018, "team": "Croatia", "squad_club_elo_rating": 1800.0},
        ]
    )

    lookup = _SCRIPT_GLOBALS["_build_team_strength_lookup"](squad_strengths, year=2018)

    assert lookup == {"France": 1900.0, "Croatia": 1800.0}


def test_apply_team_strength_adjustments_supports_secondary_column() -> None:
    adjusted = _SCRIPT_GLOBALS["_apply_team_strength_adjustments"](
        base_elo_diff=100.0,
        home_team="France",
        away_team="Norway",
        primary_lookup={"France": 1960.0, "Norway": 1820.0},
        primary_scale=0.5,
        secondary_lookup={"France": 1900.0, "Norway": 1860.0},
        secondary_scale=0.25,
    )

    assert adjusted == 180.0


def test_build_model_name_includes_secondary_component_when_present() -> None:
    model_name = _SCRIPT_GLOBALS["_build_model_name"](
        primary_rating_column="expected_xi_club_elo_rating",
        primary_scale=1.0,
        secondary_rating_column="expected_xi_goalkeeper_club_elo_rating",
        secondary_scale=0.25,
    )

    assert (
        model_name
        == "expected_xi_club_elo_rating__scale_1__expected_xi_goalkeeper_club_elo_rating__scale_0.25"
    )
