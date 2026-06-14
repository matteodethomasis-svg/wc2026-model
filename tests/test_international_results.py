from pathlib import Path

import pandas as pd

from wc2026_model.data.international_results import (
    canonicalize_team_name,
    load_international_results,
    load_scheduled_matches,
    standardize_international_results,
)


def test_standardize_international_results_creates_expected_columns() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2024-06-01",
                "home_team": "Italy",
                "away_team": "France",
                "home_score": 2,
                "away_score": 1,
                "tournament": "Friendly",
                "city": "Rome",
                "country": "Italy",
                "neutral": False,
            },
            {
                "date": "2024-06-05",
                "home_team": "Spain",
                "away_team": "Germany",
                "home_score": 1,
                "away_score": 1,
                "tournament": "FIFA World Cup qualification",
                "city": "Lisbon",
                "country": "Portugal",
                "neutral": True,
            },
        ]
    )

    standardized = standardize_international_results(raw)

    assert "match_date" in standardized.columns
    assert "home_goals" in standardized.columns
    assert "away_goals" in standardized.columns
    assert "home_result" in standardized.columns
    assert "is_competitive" in standardized.columns
    assert standardized.loc[0, "home_result"] == "home"
    assert bool(standardized.loc[0, "is_competitive"]) is False
    assert bool(standardized.loc[1, "is_competitive"]) is True


def test_standardize_international_results_drops_unplayed_future_fixtures() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2026-06-10",
                "home_team": "England",
                "away_team": "Costa Rica",
                "home_score": 3,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "London",
                "country": "England",
                "neutral": False,
            },
            {
                "date": "2026-06-25",
                "home_team": "United States",
                "away_team": "Turkey",
                "home_score": None,
                "away_score": None,
                "tournament": "FIFA World Cup",
                "city": "Inglewood",
                "country": "United States",
                "neutral": False,
            },
        ]
    )

    standardized = standardize_international_results(raw)

    assert len(standardized) == 1
    assert standardized.iloc[0]["home_team"] == "England"


def test_load_international_results_accepts_standardized_csv(tmp_path: Path) -> None:
    standardized_input = pd.DataFrame(
        [
            {
                "match_date": "2026-06-10",
                "home_team": "USA",
                "away_team": "Bosnia & Herzegovina",
                "home_goals": 3,
                "away_goals": 0,
                "tournament": "Friendlies",
                "neutral": False,
            }
        ]
    )
    csv_path = tmp_path / "standardized.csv"
    standardized_input.to_csv(csv_path, index=False)

    loaded = load_international_results(csv_path)

    assert len(loaded) == 1
    assert loaded.iloc[0]["home_team"] == "United States"
    assert loaded.iloc[0]["away_team"] == "Bosnia and Herzegovina"
    assert loaded.iloc[0]["tournament"] == "Friendly"


def test_load_scheduled_matches_filters_future_world_cup_fixtures(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2026-06-10",
                "home_team": "England",
                "away_team": "Costa Rica",
                "home_score": 3,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "London",
                "country": "England",
                "neutral": False,
            },
            {
                "date": "2026-06-25",
                "home_team": "USA",
                "away_team": "Türkiye",
                "home_score": None,
                "away_score": None,
                "tournament": "FIFA World Cup",
                "city": "Inglewood",
                "country": "United States",
                "neutral": False,
            },
            {
                "date": "2026-06-26",
                "home_team": "Japan",
                "away_team": "Sweden",
                "home_score": None,
                "away_score": None,
                "tournament": "FIFA World Cup",
                "city": "Arlington",
                "country": "United States",
                "neutral": True,
            },
        ]
    )
    csv_path = tmp_path / "fixtures.csv"
    raw.to_csv(csv_path, index=False)

    fixtures = load_scheduled_matches(
        csv_path,
        tournament="FIFA World Cup",
        start_date="2026-06-12",
    )

    assert len(fixtures) == 2
    assert fixtures.iloc[0]["home_team"] == "United States"
    assert fixtures.iloc[0]["away_team"] == "Turkey"
    assert bool(fixtures.iloc[1]["neutral"]) is True


def test_canonicalize_team_name_handles_market_and_draw_aliases() -> None:
    assert canonicalize_team_name("Côte d'Ivoire") == "Ivory Coast"
    assert canonicalize_team_name("Curaçao") == "Curaçao"
    assert canonicalize_team_name("United States of America") == "United States"
