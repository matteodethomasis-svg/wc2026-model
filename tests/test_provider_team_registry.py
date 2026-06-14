from __future__ import annotations

import pandas as pd

from wc2026_model.data import (
    build_wc2026_provider_team_registry,
    build_wc2026_provider_team_registry_summary,
    load_provider_team_registry,
    read_provider_team_ids,
)
from wc2026_model.data.provider_team_registry import (
    enrich_registry_with_api_football_feed,
    enrich_registry_with_sportmonks_feed,
)


def test_build_wc2026_provider_team_registry_uses_groups_as_primary_team_universe() -> None:
    groups = pd.DataFrame(
        [
            {"group": "I", "slot": 1, "team": "France"},
            {"group": "I", "slot": 2, "team": "Senegal"},
        ]
    )
    squads = pd.DataFrame(
        [
            {"team": "France", "player": "Mike Maignan"},
            {"team": "Norway", "player": "Erling Haaland"},
        ]
    )

    registry = build_wc2026_provider_team_registry(groups=groups, squads=squads)

    assert registry["team"].tolist() == ["France", "Senegal"]
    assert registry["group"].tolist() == ["I", "I"]
    assert registry["sportmonks_team_id"].isna().all()


def test_enrich_registry_with_provider_feeds_fills_team_ids() -> None:
    registry = build_wc2026_provider_team_registry(
        groups=pd.DataFrame(
            [
                {"group": "I", "slot": 1, "team": "France"},
                {"group": "I", "slot": 4, "team": "Norway"},
            ]
        )
    )
    sportmonks = pd.DataFrame(
        [
            {"team": "France", "team_id": 500, "source": "sportmonks_expected_lineups"},
            {"team": "Norway", "team_id": 501, "source": "sportmonks_expected_lineups"},
        ]
    )
    api_football = pd.DataFrame(
        [
            {"team": "France", "team_id": 2, "source": "api_football_injuries"},
            {"team": "Norway", "team_id": 3, "source": "api_football_injuries"},
        ]
    )

    registry = enrich_registry_with_sportmonks_feed(registry, sportmonks)
    registry = enrich_registry_with_api_football_feed(registry, api_football)

    assert registry.loc[registry["team"] == "France", "sportmonks_team_id"].iloc[0] == 500
    assert registry.loc[registry["team"] == "France", "api_football_team_id"].iloc[0] == 2
    assert registry.loc[registry["team"] == "Norway", "sportmonks_team_id"].iloc[0] == 501
    assert registry.loc[registry["team"] == "Norway", "api_football_team_id"].iloc[0] == 3


def test_registry_summary_reports_missing_provider_ids() -> None:
    registry = build_wc2026_provider_team_registry(
        groups=pd.DataFrame(
            [
                {"group": "I", "slot": 1, "team": "France"},
                {"group": "I", "slot": 2, "team": "Senegal"},
            ]
        )
    )
    sportmonks = pd.DataFrame(
        [
            {"team": "France", "team_id": 500, "source": "sportmonks_expected_lineups"},
        ]
    )

    registry = enrich_registry_with_sportmonks_feed(registry, sportmonks)
    summary = build_wc2026_provider_team_registry_summary(registry)

    assert summary["team_count"] == 2
    assert summary["sportmonks_team_ids_filled"] == 1
    assert summary["api_football_team_ids_filled"] == 0
    assert summary["missing_sportmonks_teams"] == ["Senegal"]


def test_read_provider_team_ids_reads_distinct_integers(tmp_path) -> None:
    registry = pd.DataFrame(
        [
            {"team": "France", "sportmonks_team_id": 500, "api_football_team_id": 2},
            {"team": "Senegal", "sportmonks_team_id": 501, "api_football_team_id": 7},
            {"team": "Senegal", "sportmonks_team_id": 501, "api_football_team_id": 7},
        ]
    )
    path = tmp_path / "registry.csv"
    registry.to_csv(path, index=False)

    assert read_provider_team_ids(path, provider="sportmonks") == [500, 501]
    assert read_provider_team_ids(path, provider="api_football") == [2, 7]


def test_load_provider_team_registry_preserves_existing_provider_ids(tmp_path) -> None:
    registry = pd.DataFrame(
        [
            {
                "group": "I",
                "slot": 1,
                "team": "France",
                "sportmonks_team_id": 500,
                "sportmonks_team_name": "France",
                "api_football_team_id": 2,
                "api_football_team_name": "France",
                "notes": "",
            }
        ]
    )
    path = tmp_path / "registry.csv"
    registry.to_csv(path, index=False)

    loaded = load_provider_team_registry(path)

    assert loaded.loc[0, "sportmonks_team_id"] == 500
    assert loaded.loc[0, "api_football_team_id"] == 2
