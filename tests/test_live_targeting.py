from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc2026_model.data import (
    build_upcoming_fixture_team_window,
    select_provider_team_ids_for_fixture_window,
)


def test_build_upcoming_fixture_team_window_limits_to_requested_days(tmp_path: Path) -> None:
    fixtures_path = tmp_path / "fixtures.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-12",
                "home_team": "France",
                "away_team": "Senegal",
                "home_score": None,
                "away_score": None,
                "tournament": "FIFA World Cup",
                "city": "Monterrey",
                "country": "Mexico",
                "neutral": True,
            },
            {
                "date": "2026-06-14",
                "home_team": "United States of America",
                "away_team": "Türkiye",
                "home_score": None,
                "away_score": None,
                "tournament": "FIFA World Cup",
                "city": "Kansas City",
                "country": "United States",
                "neutral": False,
            },
        ]
    ).to_csv(fixtures_path, index=False)

    window = build_upcoming_fixture_team_window(
        fixtures_path,
        tournament="FIFA World Cup",
        start_date="2026-06-12",
        window_days=2,
    )

    assert window["team"].tolist() == ["France", "Senegal"]
    assert window["opponent"].tolist() == ["Senegal", "France"]


def test_select_provider_team_ids_for_fixture_window_returns_missing_teams(tmp_path: Path) -> None:
    fixtures_path = tmp_path / "fixtures.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-12",
                "home_team": "France",
                "away_team": "Senegal",
                "home_score": None,
                "away_score": None,
                "tournament": "FIFA World Cup",
                "city": "Monterrey",
                "country": "Mexico",
                "neutral": True,
            },
            {
                "date": "2026-06-13",
                "home_team": "United States of America",
                "away_team": "Türkiye",
                "home_score": None,
                "away_score": None,
                "tournament": "FIFA World Cup",
                "city": "Kansas City",
                "country": "United States",
                "neutral": False,
            },
        ]
    ).to_csv(fixtures_path, index=False)

    registry = pd.DataFrame(
        [
            {"team": "France", "api_football_team_id": 2},
            {"team": "Senegal", "api_football_team_id": 7},
            {"team": "United States", "api_football_team_id": 2384},
            {"team": "Turkey", "api_football_team_id": None},
        ]
    )

    targeted = select_provider_team_ids_for_fixture_window(
        registry,
        provider="api_football",
        fixtures_input=fixtures_path,
        tournament="FIFA World Cup",
        start_date="2026-06-12",
        window_days=2,
    )

    assert targeted["team_ids"] == [2, 7, 2384]
    assert targeted["target_teams"] == ["France", "Senegal", "United States", "Turkey"]
    assert targeted["missing_teams"] == ["Turkey"]
    assert targeted["window_match_count"] == 2
    assert targeted["window_team_count"] == 4


def test_select_provider_team_ids_for_fixture_window_respects_max_teams(tmp_path: Path) -> None:
    fixtures_path = tmp_path / "fixtures.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-12",
                "home_team": "France",
                "away_team": "Senegal",
                "home_score": None,
                "away_score": None,
                "tournament": "FIFA World Cup",
                "city": "Monterrey",
                "country": "Mexico",
                "neutral": True,
            },
            {
                "date": "2026-06-13",
                "home_team": "United States",
                "away_team": "Turkey",
                "home_score": None,
                "away_score": None,
                "tournament": "FIFA World Cup",
                "city": "Kansas City",
                "country": "United States",
                "neutral": False,
            },
        ]
    ).to_csv(fixtures_path, index=False)

    registry = pd.DataFrame(
        [
            {"team": "France", "api_football_team_id": 2},
            {"team": "Senegal", "api_football_team_id": 7},
            {"team": "United States", "api_football_team_id": 2384},
            {"team": "Turkey", "api_football_team_id": 1118},
        ]
    )

    targeted = select_provider_team_ids_for_fixture_window(
        registry,
        provider="api_football",
        fixtures_input=fixtures_path,
        tournament="FIFA World Cup",
        start_date="2026-06-12",
        window_days=2,
        max_teams=2,
    )

    assert targeted["target_teams"] == ["France", "Senegal"]
    assert targeted["team_ids"] == [2, 7]
