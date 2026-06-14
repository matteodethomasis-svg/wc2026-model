import pandas as pd
import pytest

from wc2026_model.data import (
    build_statsbomb_team_xg_summary,
    select_statsbomb_competitions,
    standardize_statsbomb_matches,
    summarize_statsbomb_match_events,
)
from wc2026_model.features import WorldCupXGConfig, augment_with_pre_match_xg_features


def test_standardize_statsbomb_matches_maps_world_cup_payload() -> None:
    payload = [
        {
            "match_id": 3857276,
            "match_date": "2022-11-20",
            "kick_off": "17:00:00.000",
            "competition": {
                "competition_id": 43,
                "competition_name": "FIFA World Cup",
            },
            "season": {
                "season_id": 106,
                "season_name": "2022",
            },
            "home_team": {
                "home_team_name": "Qatar",
                "home_team_group": "A",
                "country": {"name": "Qatar"},
                "managers": [{"name": "Felix Sanchez"}],
            },
            "away_team": {
                "away_team_name": "Ecuador",
                "country": {"name": "Ecuador"},
                "managers": [{"name": "Gustavo Alfaro"}],
            },
            "home_score": 0,
            "away_score": 2,
            "match_week": 1,
            "competition_stage": {"name": "Group Stage"},
            "stadium": {
                "name": "Al Bayt Stadium",
                "country": {"name": "Qatar"},
            },
            "last_updated": "2026-05-04T01:46:04.664252",
        }
    ]

    standardized = standardize_statsbomb_matches(payload)

    assert len(standardized) == 1
    assert standardized.loc[0, "tournament"] == "FIFA World Cup"
    assert standardized.loc[0, "city"] == "Al Bayt Stadium"
    assert bool(standardized.loc[0, "neutral"]) is False
    assert standardized.loc[0, "home_manager"] == "Felix Sanchez"
    assert standardized.loc[0, "away_manager"] == "Gustavo Alfaro"
    assert standardized.loc[0, "source"] == "statsbomb_open_data"


def test_select_statsbomb_competitions_supports_multiple_tournaments() -> None:
    competitions = pd.DataFrame(
        [
            {
                "competition_id": 43,
                "season_id": 106,
                "competition_name": "FIFA World Cup",
                "season_name": "2022",
                "competition_gender": "male",
            },
            {
                "competition_id": 55,
                "season_id": 282,
                "competition_name": "UEFA Euro",
                "season_name": "2024",
                "competition_gender": "male",
            },
            {
                "competition_id": 223,
                "season_id": 282,
                "competition_name": "Copa America",
                "season_name": "2024",
                "competition_gender": "male",
            },
            {
                "competition_id": 53,
                "season_id": 106,
                "competition_name": "UEFA Women's Euro",
                "season_name": "2022",
                "competition_gender": "female",
            },
        ]
    )

    selected = select_statsbomb_competitions(
        competitions,
        competition_name=None,
        competition_names=["FIFA World Cup", "UEFA Euro", "Copa America"],
        competition_gender="male",
    )

    assert selected["competition_name"].tolist() == [
        "Copa America",
        "FIFA World Cup",
        "UEFA Euro",
    ]


def test_summarize_statsbomb_match_events_aggregates_xg_and_style_counts() -> None:
    events = [
        {
            "type": {"name": "Shot"},
            "team": {"name": "France"},
            "shot": {"statsbomb_xg": 0.35, "outcome": {"name": "Goal"}},
        },
        {
            "type": {"name": "Shot"},
            "team": {"name": "France"},
            "shot": {"statsbomb_xg": 0.10, "outcome": {"name": "Blocked"}},
        },
        {
            "type": {"name": "Shot"},
            "team": {"name": "Brazil"},
            "shot": {"statsbomb_xg": 0.20, "outcome": {"name": "Saved"}},
        },
        {
            "type": {"name": "Pass"},
            "team": {"name": "France"},
            "pass": {},
        },
        {
            "type": {"name": "Pass"},
            "team": {"name": "France"},
            "pass": {"outcome": {"name": "Incomplete"}},
        },
        {
            "type": {"name": "Pass"},
            "team": {"name": "Brazil"},
            "pass": {},
        },
        {
            "type": {"name": "Pressure"},
            "team": {"name": "Brazil"},
        },
    ]

    summary = summarize_statsbomb_match_events(events, home_team="France", away_team="Brazil")

    assert summary["home_xg"] == pytest.approx(0.45)
    assert summary["away_xg"] == pytest.approx(0.20)
    assert summary["home_shots"] == 2.0
    assert summary["away_shots_on_target"] == 1.0
    assert summary["home_completed_passes"] == 1.0
    assert summary["away_pass_completion_rate"] == 1.0
    assert summary["away_pressures"] == 1.0


def test_build_statsbomb_team_xg_summary_aggregates_home_and_away_rows() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_date": "2022-11-20",
                "home_team": "France",
                "away_team": "Brazil",
                "home_xg": 1.8,
                "away_xg": 1.1,
                "home_shots": 14.0,
                "away_shots": 9.0,
                "home_pressures": 18.0,
                "away_pressures": 11.0,
                "home_passes": 500.0,
                "away_passes": 430.0,
                "home_completed_passes": 450.0,
                "away_completed_passes": 380.0,
            },
            {
                "match_date": "2022-11-24",
                "home_team": "Brazil",
                "away_team": "France",
                "home_xg": 0.6,
                "away_xg": 1.4,
                "home_shots": 7.0,
                "away_shots": 12.0,
                "home_pressures": 13.0,
                "away_pressures": 20.0,
                "home_passes": 410.0,
                "away_passes": 520.0,
                "home_completed_passes": 360.0,
                "away_completed_passes": 470.0,
            },
        ]
    )

    summary = build_statsbomb_team_xg_summary(matches)
    france = summary.loc[summary["team"] == "France"].iloc[0]

    assert france["matches"] == 2
    assert france["xg_for"] == pytest.approx(3.2)
    assert france["xg_against"] == pytest.approx(1.7)
    assert france["pass_completion_rate"] == pytest.approx(920.0 / 1020.0)


def test_augment_with_pre_match_xg_features_tracks_prior_world_cup_history() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_date": "2022-11-20",
                "home_team": "France",
                "away_team": "Brazil",
                "home_xg": 1.8,
                "away_xg": 1.1,
                "home_shots": 14.0,
                "away_shots": 9.0,
            },
            {
                "match_date": "2022-11-24",
                "home_team": "France",
                "away_team": "Argentina",
                "home_xg": 0.9,
                "away_xg": 1.4,
                "home_shots": 8.0,
                "away_shots": 11.0,
            },
        ]
    )

    augmented = augment_with_pre_match_xg_features(matches, config=WorldCupXGConfig(window_size=3))

    assert pd.isna(augmented.loc[0, "home_xg_for_per_match"])
    assert augmented.loc[1, "home_xg_match_count"] == 1.0
    assert augmented.loc[1, "home_xg_for_per_match"] == 1.8
    assert augmented.loc[1, "home_shots_for_per_match"] == 14.0
