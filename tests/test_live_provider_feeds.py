from __future__ import annotations

import json

from wc2026_model.data import (
    infer_expected_lineups_provider,
    infer_injuries_provider,
    load_expected_lineups_feed,
    load_injuries_feed,
    standardize_api_football_injuries_payload,
    standardize_sportmonks_expected_lineups_payload,
)


def test_standardize_sportmonks_expected_lineups_payload_extracts_nested_fields() -> None:
    payload = {
        "data": [
            {
                "id": 11,
                "fixture_id": 19347797,
                "player_id": 101,
                "team_id": 500,
                "formation_field": "1:1",
                "formation_position": 1,
                "player_name": "Mike Maignan",
                "jersey_number": 16,
                "participant": {"id": 500, "name": "France"},
                "fixture": {"id": 19347797, "starting_at": "2026-06-16T19:00:00Z"},
                "type": {"id": 77614, "name": "Starting XI"},
                "detailed_position": {"id": 1, "name": "Goalkeeper"},
            },
            {
                "id": 12,
                "fixture_id": 19347797,
                "player_id": 202,
                "team_id": 500,
                "player_name": "Bradley Barcola",
                "participant": {"id": 500, "name": "France"},
                "fixture": {"id": 19347797, "starting_at": "2026-06-16T19:00:00Z"},
                "type": {"id": 77615, "name": "Bench"},
            },
        ]
    }

    frame = standardize_sportmonks_expected_lineups_payload(payload)

    assert frame.shape[0] == 2
    assert frame.loc[0, "team"] == "France"
    assert frame.loc[0, "player"] == "Mike Maignan"
    assert frame.loc[0, "position"] == "GK"
    assert bool(frame.loc[0, "is_expected_starter"]) is True
    assert bool(frame.loc[1, "is_expected_starter"]) is False
    assert frame.loc[0, "source"] == "sportmonks_expected_lineups"


def test_standardize_api_football_injuries_payload_extracts_nested_fields() -> None:
    payload = {
        "response": [
            {
                "team": {"id": 2, "name": "France"},
                "player": {
                    "id": 101,
                    "name": "Mike Maignan",
                    "type": "Injury",
                    "reason": "Hamstring",
                },
                "fixture": {"id": 5001, "date": "2026-06-16T19:00:00+00:00"},
                "league": {"id": 1, "name": "FIFA World Cup"},
            },
            {
                "team": {"id": 3, "name": "Norway"},
                "player": {
                    "id": 102,
                    "name": "Martin Odegaard",
                    "type": "Doubtful",
                    "reason": "Knock",
                },
                "fixture": {"id": 5002, "date": "2026-06-16T22:00:00+00:00"},
                "league": {"id": 1, "name": "FIFA World Cup"},
            },
        ]
    }

    frame = standardize_api_football_injuries_payload(payload)

    assert frame.shape[0] == 2
    assert frame.loc[0, "team"] == "France"
    assert frame.loc[0, "player"] == "Mike Maignan"
    assert frame.loc[0, "availability_status"] == "unavailable"
    assert frame.loc[1, "availability_status"] == "doubtful"
    assert frame.loc[0, "source"] == "api_football_injuries"


def test_standardize_api_football_injuries_payload_preserves_schema_when_empty() -> None:
    frame = standardize_api_football_injuries_payload({"response": []})

    assert frame.empty
    assert frame.columns.tolist() == [
        "fixture_id",
        "match_date",
        "team",
        "player",
        "status",
        "reason",
        "availability_status",
        "report_date",
        "expected_return_date",
        "team_id",
        "player_id",
        "league_id",
        "league_name",
        "source",
    ]


def test_load_expected_lineups_feed_auto_detects_sportmonks_json(tmp_path) -> None:
    payload = {
        "data": [
            {
                "id": 11,
                "fixture_id": 19347797,
                "player_id": 101,
                "team_id": 500,
                "player_name": "Mike Maignan",
                "participant": {"id": 500, "name": "France"},
                "fixture": {"id": 19347797, "starting_at": "2026-06-16T19:00:00Z"},
                "type": {"id": 77614, "name": "Starting XI"},
            }
        ]
    }
    path = tmp_path / "sportmonks_expected_lineups.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert infer_expected_lineups_provider(payload) == "sportmonks"
    frame = load_expected_lineups_feed(path)

    assert frame.shape[0] == 1
    assert frame.loc[0, "team"] == "France"
    assert frame.loc[0, "player"] == "Mike Maignan"


def test_load_injuries_feed_auto_detects_api_football_json(tmp_path) -> None:
    payload = {
        "response": [
            {
                "team": {"id": 2, "name": "France"},
                "player": {
                    "id": 101,
                    "name": "Mike Maignan",
                    "type": "Injury",
                    "reason": "Hamstring",
                },
                "fixture": {"id": 5001, "date": "2026-06-16T19:00:00+00:00"},
                "league": {"id": 1, "name": "FIFA World Cup"},
            }
        ]
    }
    path = tmp_path / "api_football_injuries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert infer_injuries_provider(payload) == "api_football"
    frame = load_injuries_feed(path)

    assert frame.shape[0] == 1
    assert frame.loc[0, "team"] == "France"
    assert frame.loc[0, "availability_status"] == "unavailable"
