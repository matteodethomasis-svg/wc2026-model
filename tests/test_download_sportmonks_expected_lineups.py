from __future__ import annotations

import runpy

import pandas as pd


_SCRIPT_GLOBALS = runpy.run_path("scripts/download_sportmonks_expected_lineups.py")


def test_resolve_team_ids_supports_provider_registry(tmp_path) -> None:
    registry = pd.DataFrame(
        [
            {"team": "France", "sportmonks_team_id": 500},
            {"team": "Senegal", "sportmonks_team_id": 501},
        ]
    )
    registry_path = tmp_path / "registry.csv"
    registry.to_csv(registry_path, index=False)

    resolved = _SCRIPT_GLOBALS["_resolve_team_ids"](
        csv_team_ids=None,
        registry_input=registry_path,
        team_ids_input=None,
    )

    assert resolved == [500, 501]
