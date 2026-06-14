import pandas as pd
import pytest

from wc2026_model.features import (
    attach_confederation_features,
    attach_fixture_h2h_features,
    augment_with_pre_match_h2h_features,
    get_team_confederation,
)


def test_get_team_confederation_maps_known_teams() -> None:
    assert get_team_confederation("France") == "UEFA"
    assert get_team_confederation("Japan") == "AFC"
    assert get_team_confederation("United States") == "CONCACAF"


def test_attach_confederation_features_builds_flags() -> None:
    frame = pd.DataFrame(
        [
            {
                "home_team": "France",
                "away_team": "Japan",
            },
            {
                "home_team": "Brazil",
                "away_team": "Argentina",
            },
        ]
    )

    enriched = attach_confederation_features(frame)

    assert enriched.loc[0, "home_confederation"] == "UEFA"
    assert enriched.loc[0, "away_confederation"] == "AFC"
    assert enriched.loc[0, "same_confederation"] == pytest.approx(0.0)
    assert enriched.loc[1, "same_confederation"] == pytest.approx(1.0)
    assert enriched.loc[0, "home_is_uefa"] == pytest.approx(1.0)
    assert enriched.loc[0, "away_is_afc"] == pytest.approx(1.0)


def test_augment_with_pre_match_h2h_features_is_directional_and_pre_match() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_date": "2024-01-01",
                "home_team": "France",
                "away_team": "Brazil",
                "home_goals": 2,
                "away_goals": 1,
            },
            {
                "match_date": "2024-01-10",
                "home_team": "Brazil",
                "away_team": "France",
                "home_goals": 0,
                "away_goals": 0,
            },
            {
                "match_date": "2024-01-20",
                "home_team": "France",
                "away_team": "Brazil",
                "home_goals": 1,
                "away_goals": 3,
            },
        ]
    )

    augmented = augment_with_pre_match_h2h_features(matches)

    assert augmented.loc[0, "h2h_match_count"] == pytest.approx(0.0)
    assert pd.isna(augmented.loc[0, "h2h_home_win_rate"])
    assert augmented.loc[1, "h2h_match_count"] == pytest.approx(1.0)
    assert augmented.loc[1, "h2h_away_win_rate"] == pytest.approx(1.0)
    assert augmented.loc[2, "h2h_match_count"] == pytest.approx(2.0)
    assert augmented.loc[2, "h2h_home_win_rate"] == pytest.approx(0.5)
    assert augmented.loc[2, "h2h_draw_rate"] == pytest.approx(0.5)


def test_attach_fixture_h2h_features_uses_latest_history() -> None:
    historical_matches = pd.DataFrame(
        [
            {
                "match_date": "2024-01-01",
                "home_team": "France",
                "away_team": "Brazil",
                "home_goals": 2,
                "away_goals": 1,
            },
            {
                "match_date": "2024-01-10",
                "home_team": "Brazil",
                "away_team": "France",
                "home_goals": 0,
                "away_goals": 0,
            },
        ]
    )
    fixtures = pd.DataFrame(
        [
            {
                "match_date": "2024-02-01",
                "home_team": "France",
                "away_team": "Brazil",
            }
        ]
    )

    enriched = attach_fixture_h2h_features(fixtures, historical_matches)

    assert enriched.loc[0, "h2h_match_count"] == pytest.approx(2.0)
    assert enriched.loc[0, "h2h_home_win_rate"] == pytest.approx(0.5)
    assert enriched.loc[0, "h2h_draw_rate"] == pytest.approx(0.5)
