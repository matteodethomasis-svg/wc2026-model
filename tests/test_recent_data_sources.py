import pandas as pd

from wc2026_model.data import (
    combine_standardized_results,
    standardize_cup26_open_results,
    standardize_football_data_payloads,
)
from wc2026_model.data.international_results import (
    canonicalize_team_name,
    canonicalize_tournament_name,
)


def test_canonicalize_tournament_name_normalizes_recent_source_labels() -> None:
    assert canonicalize_tournament_name("World Cup - Qualification Asia") == (
        "FIFA World Cup qualification"
    )
    assert canonicalize_tournament_name("Euro Championship - Qualification") == (
        "UEFA Euro qualification"
    )
    assert canonicalize_tournament_name("Friendlies") == "Friendly"
    assert canonicalize_team_name("Bosnia & Herzegovina") == "Bosnia and Herzegovina"
    assert canonicalize_team_name("USA") == "United States"


def test_standardize_cup26_open_results_maps_recent_json_payload() -> None:
    payload = {
        "generatedAt": "2026-06-11T10:01:41.686Z",
        "matches": [
            {
                "id": 1,
                "date": "2026-06-10",
                "homeName": "England",
                "awayName": "Costa Rica",
                "hg": 3,
                "ag": 0,
                "leagueId": 10,
                "leagueName": "Friendlies",
            },
            {
                "id": 2,
                "date": "2026-06-08",
                "homeName": "Spain",
                "awayName": "Germany",
                "hg": 2,
                "ag": 1,
                "leagueId": 111,
                "leagueName": "Euro Championship",
            },
        ],
    }

    standardized = standardize_cup26_open_results(payload)

    assert list(standardized["tournament"]) == ["UEFA Euro", "Friendly"]
    assert bool(standardized.loc[0, "neutral"]) is True
    assert bool(standardized.loc[1, "neutral"]) is False
    assert set(standardized["source"]) == {"cup26_open"}
    assert standardized["source_generated_at_utc"].nunique() == 1


def test_standardize_football_data_payloads_maps_api_response() -> None:
    payload = {
        "area": {"name": "World"},
        "competition": {"code": "WC", "name": "FIFA World Cup"},
        "matches": [
            {
                "id": 42,
                "utcDate": "2026-06-15T19:00:00Z",
                "homeTeam": {"name": "Brazil"},
                "awayTeam": {"name": "Japan"},
                "venue": "MetLife Stadium",
                "lastUpdated": "2026-06-15T21:00:00Z",
                "score": {
                    "regularTime": {"home": 2, "away": 1},
                    "fullTime": {"home": 2, "away": 1},
                },
            }
        ],
    }

    standardized = standardize_football_data_payloads([payload])

    assert len(standardized) == 1
    assert standardized.loc[0, "tournament"] == "FIFA World Cup"
    assert standardized.loc[0, "city"] == "MetLife Stadium"
    assert bool(standardized.loc[0, "neutral"]) is True
    assert standardized.loc[0, "source"] == "football_data_api"
    assert standardized.loc[0, "source_match_id"] == 42


def test_combine_standardized_results_prefers_higher_priority_source() -> None:
    historical = pd.DataFrame(
        [
            {
                "match_date": pd.Timestamp("2026-06-10"),
                "home_team": "England",
                "away_team": "Costa Rica",
                "home_goals": 3,
                "away_goals": 0,
                "tournament": "Friendly",
                "source": "historical_csv",
            }
        ]
    )
    cup26_open = pd.DataFrame(
        [
            {
                "match_date": pd.Timestamp("2026-06-10"),
                "home_team": "England",
                "away_team": "Costa Rica",
                "home_goals": 3,
                "away_goals": 0,
                "tournament": "Friendly",
                "source": "cup26_open",
                "source_generated_at_utc": "2026-06-11T10:01:41.686Z",
            }
        ]
    )

    combined = combine_standardized_results([historical, cup26_open])

    assert len(combined) == 1
    assert combined.loc[0, "source"] == "cup26_open"


def test_combine_standardized_results_dedupes_same_fixture_logged_one_day_apart() -> None:
    # Two sources logged the same friendly a day apart (timezone / record date).
    # Exact-date dedupe misses it; this must still collapse to a single match,
    # otherwise the result is double-counted in the Elo (the Argentina inflation bug).
    historical = pd.DataFrame(
        [
            {
                "match_date": pd.Timestamp("2025-10-14"),
                "home_team": "Puerto Rico",
                "away_team": "Argentina",
                "home_goals": 0,
                "away_goals": 6,
                "tournament": "Friendly",
                "source": "historical_csv",
            }
        ]
    )
    cup26_open = pd.DataFrame(
        [
            {
                "match_date": pd.Timestamp("2025-10-15"),
                "home_team": "Puerto Rico",
                "away_team": "Argentina",
                "home_goals": 0,
                "away_goals": 6,
                "tournament": "Friendly",
                "source": "cup26_open",
            }
        ]
    )

    combined = combine_standardized_results([historical, cup26_open])

    assert len(combined) == 1
    assert combined.loc[0, "source"] == "cup26_open"


def test_combine_standardized_results_keeps_distinct_back_to_back_fixtures() -> None:
    # Same source, same matchup/score on consecutive days = a real two-leg friendly
    # set; must NOT be collapsed. Guards against the dedupe being too aggressive.
    frame = pd.DataFrame(
        [
            {
                "match_date": pd.Timestamp("2025-10-10"),
                "home_team": "Argentina",
                "away_team": "Venezuela",
                "home_goals": 1,
                "away_goals": 0,
                "tournament": "Friendly",
                "source": "cup26_open",
            },
            {
                "match_date": pd.Timestamp("2025-10-13"),
                "home_team": "Argentina",
                "away_team": "Venezuela",
                "home_goals": 1,
                "away_goals": 0,
                "tournament": "Friendly",
                "source": "cup26_open",
            },
        ]
    )

    combined = combine_standardized_results([frame])

    # 3 days apart > tolerance, so both survive.
    assert len(combined) == 2
