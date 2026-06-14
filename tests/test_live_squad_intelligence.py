from __future__ import annotations

import json

import pandas as pd

from wc2026_model.data import (
    aggregate_team_availability_features,
    build_live_squad_intelligence,
    build_live_squad_intelligence_summary,
    load_flat_file_table,
    normalize_person_name,
    standardize_official_squads,
)


def test_standardize_official_squads_normalizes_team_and_player_keys() -> None:
    official = pd.DataFrame(
        [
            {"team": "Bosnia & Herzegovina", "player": "Kylian Mbappé", "position": "FW"},
            {"team": "USA", "player": "Matt Turner", "position": "GK"},
        ]
    )

    standardized = standardize_official_squads(official)

    assert standardized["team"].tolist() == ["Bosnia and Herzegovina", "United States"]
    assert standardized["player_key"].tolist() == ["kylian mbappe", "matt turner"]
    assert normalize_person_name("Míke Maignan") == "mike maignan"


def test_build_live_squad_intelligence_merges_lineups_and_injuries() -> None:
    official = pd.DataFrame(
        [
            {"team": "France", "player": "Mike Maignan", "position": "GK"},
            {"team": "France", "player": "Kylian Mbappe", "position": "FW"},
            {"team": "France", "player": "Aurelien Tchouameni", "position": "MF"},
            {"team": "Norway", "player": "Orjan Nyland", "position": "GK"},
            {"team": "Norway", "player": "Erling Haaland", "position": "FW"},
            {"team": "Norway", "player": "Martin Odegaard", "position": "MF"},
        ]
    )
    expected_lineups = pd.DataFrame(
        [
            {
                "fixture_id": "wc26-001",
                "match_date": "2026-06-16",
                "team": "France",
                "player": "Mike Maignan",
                "position": "GK",
                "formation": "4-3-3",
                "is_expected_starter": True,
                "lineup_confidence": 0.9,
            },
            {
                "fixture_id": "wc26-001",
                "match_date": "2026-06-16",
                "team": "France",
                "player": "Kylian Mbappé",
                "position": "FW",
                "formation": "4-3-3",
                "is_expected_starter": True,
                "lineup_confidence": 0.9,
            },
            {
                "fixture_id": "wc26-001",
                "match_date": "2026-06-16",
                "team": "Norway",
                "player": "Orjan Nyland",
                "position": "GK",
                "formation": "4-4-2",
                "is_expected_starter": True,
                "lineup_confidence": 0.8,
            },
            {
                "fixture_id": "wc26-001",
                "match_date": "2026-06-16",
                "team": "Norway",
                "player": "Erling Haaland",
                "position": "FW",
                "formation": "4-4-2",
                "is_expected_starter": True,
                "lineup_confidence": 0.8,
            },
            {
                "fixture_id": "wc26-001",
                "match_date": "2026-06-16",
                "team": "France",
                "player": "Unknown Callup",
                "position": "DF",
                "formation": "4-3-3",
                "is_expected_starter": True,
                "lineup_confidence": 0.6,
            },
        ]
    )
    injuries = pd.DataFrame(
        [
            {"team": "France", "player": "Mike Maignan", "status": "Out", "reason": "hamstring"},
            {"team": "Norway", "player": "Martin Odegaard", "status": "Doubtful", "reason": "knock"},
        ]
    )

    player_frame = build_live_squad_intelligence(
        official,
        expected_lineups=expected_lineups,
        injuries=injuries,
    )
    team_features = aggregate_team_availability_features(player_frame)
    summary = build_live_squad_intelligence_summary(
        player_frame,
        expected_lineups=expected_lineups,
        injuries=injuries,
    )

    france_maignan = player_frame.loc[
        (player_frame["team"] == "France") & (player_frame["player"] == "Mike Maignan")
    ].iloc[0]
    assert bool(france_maignan["is_expected_starter"]) is True
    assert france_maignan["availability_status"] == "unavailable"
    assert bool(france_maignan["is_unavailable"]) is True

    norway_odegaard = player_frame.loc[
        (player_frame["team"] == "Norway") & (player_frame["player"] == "Martin Odegaard")
    ].iloc[0]
    assert norway_odegaard["availability_status"] == "doubtful"
    assert bool(norway_odegaard["is_doubtful"]) is True

    france_features = team_features.loc[team_features["team"] == "France"].iloc[0]
    assert int(france_features["expected_starter_count"]) == 2
    assert int(france_features["unavailable_expected_starter_count"]) == 1
    assert bool(france_features["goalkeeper_starter_available"]) is False

    norway_features = team_features.loc[team_features["team"] == "Norway"].iloc[0]
    assert int(norway_features["expected_starter_count"]) == 2
    assert int(norway_features["doubtful_expected_starter_count"]) == 0
    assert bool(norway_features["goalkeeper_starter_available"]) is True

    assert summary["expected_lineup_unmatched_rows"] == 1
    assert summary["injury_unmatched_rows"] == 0


def test_load_flat_file_table_reads_json_response_wrapper(tmp_path) -> None:
    payload = {
        "response": [
            {"team": "France", "player": "Mike Maignan", "status": "Out"},
            {"team": "France", "player": "Kylian Mbappe", "status": "Available"},
        ]
    }
    path = tmp_path / "injuries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_flat_file_table(path)

    assert loaded.shape == (2, 3)
    assert loaded["player"].tolist() == ["Mike Maignan", "Kylian Mbappe"]


def test_aggregate_team_availability_features_defaults_lineup_confidence_to_zero() -> None:
    official = pd.DataFrame(
        [
            {"team": "France", "player": "Mike Maignan", "position": "GK"},
            {"team": "France", "player": "Kylian Mbappe", "position": "FW"},
        ]
    )

    player_frame = build_live_squad_intelligence(official)
    team_features = aggregate_team_availability_features(player_frame)

    assert float(team_features.loc[0, "lineup_confidence"]) == 0.0
