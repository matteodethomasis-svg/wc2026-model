import pandas as pd
import pytest

from wc2026_model.features import (
    WorldCupXGConfig,
    attach_latest_team_xg_features,
    build_latest_team_xg_snapshot,
)


def test_build_latest_team_xg_snapshot_uses_recent_window() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_date": "2024-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_xg": 1.2,
                "away_xg": 0.8,
                "home_shots": 10.0,
                "away_shots": 7.0,
                "home_shots_on_target": 4.0,
                "away_shots_on_target": 2.0,
            },
            {
                "match_date": "2024-01-08",
                "home_team": "Gamma",
                "away_team": "Alpha",
                "home_xg": 0.5,
                "away_xg": 1.6,
                "home_shots": 5.0,
                "away_shots": 12.0,
                "home_shots_on_target": 1.0,
                "away_shots_on_target": 5.0,
            },
            {
                "match_date": "2024-01-16",
                "home_team": "Alpha",
                "away_team": "Delta",
                "home_xg": 0.9,
                "away_xg": 1.0,
                "home_shots": 8.0,
                "away_shots": 9.0,
                "home_shots_on_target": 3.0,
                "away_shots_on_target": 4.0,
            },
        ]
    )

    snapshot = build_latest_team_xg_snapshot(matches, config=WorldCupXGConfig(window_size=2))
    alpha = snapshot.loc[snapshot["team"] == "Alpha"].iloc[0]

    assert alpha["xg_match_count"] == pytest.approx(2.0)
    assert alpha["xg_for_per_match"] == pytest.approx((1.6 + 0.9) / 2.0)
    assert alpha["shots_for_per_match"] == pytest.approx((12.0 + 8.0) / 2.0)
    assert alpha["shots_on_target_for_per_match"] == pytest.approx((5.0 + 3.0) / 2.0)
    assert alpha["shot_accuracy_for"] == pytest.approx(8.0 / 20.0)


def test_attach_latest_team_xg_features_merges_home_and_away_profiles() -> None:
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "1",
                "home_team": "Alpha",
                "away_team": "Beta",
            }
        ]
    )
    snapshot = pd.DataFrame(
        [
            {
                "team": "Alpha",
                "xg_match_count": 3.0,
                "xg_for_per_match": 1.4,
            },
            {
                "team": "Beta",
                "xg_match_count": 2.0,
                "xg_for_per_match": 0.9,
            },
        ]
    )

    enriched = attach_latest_team_xg_features(fixtures, snapshot)

    assert enriched.loc[0, "home_xg_match_count"] == pytest.approx(3.0)
    assert enriched.loc[0, "away_xg_match_count"] == pytest.approx(2.0)
    assert enriched.loc[0, "home_xg_for_per_match"] == pytest.approx(1.4)
    assert enriched.loc[0, "away_xg_for_per_match"] == pytest.approx(0.9)
