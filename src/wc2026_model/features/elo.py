from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloConfig:
    initial_rating: float = 1500.0
    home_field_advantage: float = 100.0
    # Friendlies are weak strength signals (rotated squads, low stakes), so they move
    # the Elo less. A historical backtest on competitive holdout matches showed log
    # loss improves monotonically as the friendly weight drops from 20; K=10 captures
    # most of that gain while still letting informative friendlies (e.g. a heavy loss)
    # register. Competitive weights come from infer_match_importance (40-60).
    base_k_friendly: float = 10.0
    base_k_competitive: float = 30.0


def infer_match_importance(tournament_name: str, config: EloConfig | None = None) -> float:
    config = config or EloConfig()
    normalized = tournament_name.lower()

    if "world cup qualification" in normalized:
        return 40.0
    if normalized == "fifa world cup":
        return 60.0
    if "qualification" in normalized:
        return 40.0
    if "nations league" in normalized:
        return 35.0
    if any(
        token in normalized
        for token in (
            "uefa euro",
            "copa américa",
            "copa america",
            "african cup of nations",
            "asian cup",
            "gold cup",
        )
    ):
        return 50.0
    if "friendly" in normalized:
        return config.base_k_friendly
    return config.base_k_competitive


def goal_difference_multiplier(goal_difference: int) -> float:
    absolute_difference = abs(goal_difference)
    if absolute_difference <= 1:
        return 1.0
    if absolute_difference == 2:
        return 1.5
    return (11.0 + absolute_difference) / 8.0


def augment_with_pre_match_elo(
    matches: pd.DataFrame,
    *,
    config: EloConfig | None = None,
) -> pd.DataFrame:
    config = config or EloConfig()
    required_columns = {
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "tournament",
        "neutral",
    }
    missing_columns = required_columns.difference(matches.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns for Elo augmentation: {missing}")

    dataframe = matches.sort_values(
        ["match_date", "home_team", "away_team"], kind="stable"
    ).reset_index(drop=True)
    ratings: dict[str, float] = {}

    home_elo_pre: list[float] = []
    away_elo_pre: list[float] = []
    expected_home_result: list[float] = []
    elo_rating_delta: list[float] = []
    match_importance: list[float] = []

    for row in dataframe.itertuples(index=False):
        home_team = row.home_team
        away_team = row.away_team
        home_rating = ratings.get(home_team, config.initial_rating)
        away_rating = ratings.get(away_team, config.initial_rating)

        home_elo_pre.append(home_rating)
        away_elo_pre.append(away_rating)

        home_advantage = 0.0 if row.neutral else config.home_field_advantage
        rating_difference = (home_rating + home_advantage) - away_rating
        expected_home = 1.0 / (1.0 + 10.0 ** (-rating_difference / 400.0))
        expected_home_result.append(expected_home)

        if row.home_goals > row.away_goals:
            actual_home = 1.0
        elif row.home_goals == row.away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0

        importance = infer_match_importance(row.tournament, config)
        multiplier = goal_difference_multiplier(row.home_goals - row.away_goals)
        delta = importance * multiplier * (actual_home - expected_home)

        ratings[home_team] = home_rating + delta
        ratings[away_team] = away_rating - delta

        elo_rating_delta.append(delta)
        match_importance.append(importance)

    dataframe["home_elo_pre"] = home_elo_pre
    dataframe["away_elo_pre"] = away_elo_pre
    dataframe["elo_diff_pre"] = dataframe["home_elo_pre"] - dataframe["away_elo_pre"]
    dataframe["elo_expected_home"] = expected_home_result
    dataframe["elo_rating_delta_home"] = elo_rating_delta
    dataframe["match_importance"] = match_importance
    return dataframe


def build_latest_elo_ratings(
    matches: pd.DataFrame,
    *,
    config: EloConfig | None = None,
) -> pd.DataFrame:
    augmented = augment_with_pre_match_elo(matches, config=config).copy()
    if augmented.empty:
        return pd.DataFrame(
            columns=["team", "elo_rating", "last_match_date", "matches_played"]
        )

    augmented["match_sequence"] = np.arange(len(augmented), dtype=int)

    home_updates = (
        augmented[
            [
                "match_date",
                "match_sequence",
                "home_team",
                "home_elo_pre",
                "elo_rating_delta_home",
            ]
        ]
        .rename(
            columns={
                "home_team": "team",
                "home_elo_pre": "elo_pre",
                "elo_rating_delta_home": "elo_delta",
            }
        )
        .assign(elo_rating=lambda frame: frame["elo_pre"] + frame["elo_delta"])
    )
    away_updates = (
        augmented[
            [
                "match_date",
                "match_sequence",
                "away_team",
                "away_elo_pre",
                "elo_rating_delta_home",
            ]
        ]
        .rename(
            columns={
                "away_team": "team",
                "away_elo_pre": "elo_pre",
                "elo_rating_delta_home": "elo_delta",
            }
        )
        .assign(elo_rating=lambda frame: frame["elo_pre"] - frame["elo_delta"])
    )

    rating_updates = pd.concat([home_updates, away_updates], ignore_index=True)
    latest_ratings = (
        rating_updates.sort_values(
            ["match_date", "match_sequence", "team"], kind="stable"
        )
        .drop_duplicates(subset="team", keep="last")
        .loc[:, ["team", "elo_rating", "match_date"]]
        .rename(columns={"match_date": "last_match_date"})
    )

    matches_played = (
        pd.concat(
            [
                augmented["home_team"].rename("team"),
                augmented["away_team"].rename("team"),
            ],
            axis=0,
        )
        .value_counts()
        .rename_axis("team")
        .reset_index(name="matches_played")
    )
    latest_ratings = latest_ratings.merge(matches_played, on="team", how="left")
    latest_ratings["matches_played"] = latest_ratings["matches_played"].fillna(0).astype(int)
    return latest_ratings.sort_values("elo_rating", ascending=False, kind="stable").reset_index(
        drop=True
    )
