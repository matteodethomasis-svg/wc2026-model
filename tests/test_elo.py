import pandas as pd

from wc2026_model.data.international_results import standardize_international_results
from wc2026_model.features.elo import augment_with_pre_match_elo


def test_augment_with_pre_match_elo_starts_at_initial_ratings() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2024-06-01",
                "home_team": "Italy",
                "away_team": "France",
                "home_score": 2,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "Rome",
                "country": "Italy",
                "neutral": False,
            },
            {
                "date": "2024-06-05",
                "home_team": "Italy",
                "away_team": "Spain",
                "home_score": 1,
                "away_score": 1,
                "tournament": "Friendly",
                "city": "Milan",
                "country": "Italy",
                "neutral": False,
            },
        ]
    )

    standardized = standardize_international_results(raw)
    enriched = augment_with_pre_match_elo(standardized)

    assert enriched.loc[0, "home_elo_pre"] == 1500.0
    assert enriched.loc[0, "away_elo_pre"] == 1500.0
    assert enriched.loc[1, "home_elo_pre"] > 1500.0
    assert enriched.loc[1, "away_elo_pre"] == 1500.0
