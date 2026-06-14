from __future__ import annotations

import runpy

import pandas as pd
import pytest


_SCRIPT_GLOBALS = runpy.run_path("scripts/download_api_football_injuries.py")


def test_resolve_team_ids_supports_provider_registry(tmp_path) -> None:
    registry = pd.DataFrame(
        [
            {"team": "France", "api_football_team_id": 2},
            {"team": "Senegal", "api_football_team_id": 7},
        ]
    )
    registry_path = tmp_path / "registry.csv"
    registry.to_csv(registry_path, index=False)

    resolved = _SCRIPT_GLOBALS["_resolve_team_ids"](
        csv_team_ids=None,
        registry_input=registry_path,
    )

    assert resolved == [2, 7]


def test_apply_fixture_window_targeting_uses_upcoming_wc_teams(tmp_path) -> None:
    registry = pd.DataFrame(
        [
            {"team": "France", "api_football_team_id": 2},
            {"team": "Senegal", "api_football_team_id": 7},
            {"team": "United States", "api_football_team_id": 2384},
            {"team": "Turkey", "api_football_team_id": 1118},
        ]
    )
    registry_path = tmp_path / "registry.csv"
    registry.to_csv(registry_path, index=False)

    fixtures = pd.DataFrame(
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
    )
    fixtures_path = tmp_path / "fixtures.csv"
    fixtures.to_csv(fixtures_path, index=False)

    targeted = _SCRIPT_GLOBALS["_apply_fixture_window_targeting"](
        team_ids=[],
        registry_input=registry_path,
        fixtures_input=fixtures_path,
        tournament="FIFA World Cup",
        start_date="2026-06-12",
        upcoming_window_days=2,
        max_teams=16,
    )

    assert targeted["team_ids"] == [2, 7]
    assert targeted["targeting_summary"]["target_teams"] == ["France", "Senegal"]


def test_apply_fixture_window_targeting_can_leave_empty_ids_when_registry_is_unmapped(tmp_path) -> None:
    registry = pd.DataFrame(
        [
            {"team": "France", "api_football_team_id": None},
            {"team": "Senegal", "api_football_team_id": None},
        ]
    )
    registry_path = tmp_path / "registry.csv"
    registry.to_csv(registry_path, index=False)

    fixtures = pd.DataFrame(
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
            }
        ]
    )
    fixtures_path = tmp_path / "fixtures.csv"
    fixtures.to_csv(fixtures_path, index=False)

    targeted = _SCRIPT_GLOBALS["_apply_fixture_window_targeting"](
        team_ids=[],
        registry_input=registry_path,
        fixtures_input=fixtures_path,
        tournament="FIFA World Cup",
        start_date="2026-06-12",
        upcoming_window_days=2,
        max_teams=16,
    )

    assert targeted["team_ids"] == []
    assert targeted["targeting_summary"]["missing_teams"] == ["France", "Senegal"]
