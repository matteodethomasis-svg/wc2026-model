from __future__ import annotations

import runpy
from pathlib import Path

import pandas as pd


_SCRIPT_GLOBALS = runpy.run_path("scripts/run_wc2026_live_pipeline.py")


def test_resolve_provider_team_ids_merges_manual_and_registry_ids() -> None:
    registry = pd.DataFrame(
        [
            {"team": "France", "sportmonks_team_id": 500, "api_football_team_id": 2},
            {"team": "Senegal", "sportmonks_team_id": 501, "api_football_team_id": 7},
            {"team": "Norway", "sportmonks_team_id": 500, "api_football_team_id": None},
        ]
    )

    sportmonks_ids = _SCRIPT_GLOBALS["_resolve_provider_team_ids"](
        csv_team_ids="101, 102, 101",
        registry=registry,
        provider="sportmonks",
    )
    api_football_ids = _SCRIPT_GLOBALS["_resolve_provider_team_ids"](
        csv_team_ids=None,
        registry=registry,
        provider="api_football",
    )

    assert sportmonks_ids == [101, 102, 500, 501]
    assert api_football_ids == [2, 7]


def test_resolve_provider_team_ids_can_skip_registry_ids() -> None:
    registry = pd.DataFrame(
        [
            {"team": "France", "api_football_team_id": 2},
            {"team": "Senegal", "api_football_team_id": 7},
        ]
    )

    api_football_ids = _SCRIPT_GLOBALS["_resolve_provider_team_ids"](
        csv_team_ids="101, 102",
        registry=registry,
        provider="api_football",
        include_registry=False,
    )

    assert api_football_ids == [101, 102]


def test_merge_targeted_team_ids_uses_only_window_ids_for_free_plan_default() -> None:
    merged = _SCRIPT_GLOBALS["_merge_targeted_team_ids"](
        base_team_ids=[2, 7, 15],
        targeted_team_ids=[5529, 1113, 2384],
        free_plan=True,
        explicit_csv_team_ids=None,
    )

    assert merged == [5529, 1113, 2384]


def test_merge_targeted_team_ids_keeps_manual_ids_when_explicitly_supplied() -> None:
    merged = _SCRIPT_GLOBALS["_merge_targeted_team_ids"](
        base_team_ids=[9001, 9002],
        targeted_team_ids=[5529, 1113],
        free_plan=True,
        explicit_csv_team_ids="9001,9002",
    )

    assert merged == [5529, 1113, 9001, 9002]


def test_resolve_existing_feed_path_accepts_csv_fallback(tmp_path: Path) -> None:
    csv_path = tmp_path / "expected_lineups.csv"
    csv_path.write_text("team,team_id\nFrance,500\n", encoding="utf-8")

    resolved = _SCRIPT_GLOBALS["_resolve_existing_feed_path"](
        None,
        tmp_path / "missing.json",
        csv_path,
    )

    assert resolved == csv_path
