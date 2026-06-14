from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wc2026_model.features import EloConfig, augment_with_pre_match_elo
from wc2026_model.models import DixonColesModel, exponential_time_decay_weights


@dataclass(frozen=True)
class BaselineTrainingConfig:
    min_match_date: str | None = "2010-01-01"
    training_cutoff: str | None = None
    min_team_matches: int = 5
    time_decay_xi: float = 0.001
    l2_penalty: float = 0.01
    maxiter: int = 1000
    elo_config: EloConfig = EloConfig()


def build_training_frame(
    results: pd.DataFrame,
    *,
    config: BaselineTrainingConfig | None = None,
) -> pd.DataFrame:
    config = config or BaselineTrainingConfig()
    dataframe = results.copy()
    dataframe["match_date"] = pd.to_datetime(dataframe["match_date"], errors="raise")

    if config.min_match_date is not None:
        dataframe = dataframe[dataframe["match_date"] >= pd.Timestamp(config.min_match_date)]
    if config.training_cutoff is not None:
        dataframe = dataframe[dataframe["match_date"] < pd.Timestamp(config.training_cutoff)]

    dataframe = dataframe.sort_values(
        ["match_date", "home_team", "away_team"], kind="stable"
    ).reset_index(drop=True)
    dataframe = augment_with_pre_match_elo(dataframe, config=config.elo_config)

    team_match_counts = (
        pd.concat(
            [
                dataframe["home_team"].rename("team"),
                dataframe["away_team"].rename("team"),
            ],
            axis=0,
        )
        .value_counts()
        .rename_axis("team")
        .reset_index(name="match_count")
    )
    eligible_teams = set(
        team_match_counts.loc[
            team_match_counts["match_count"] >= config.min_team_matches, "team"
        ]
    )
    filtered = dataframe[
        dataframe["home_team"].isin(eligible_teams) & dataframe["away_team"].isin(eligible_teams)
    ].copy()
    filtered["sample_weight"] = exponential_time_decay_weights(
        filtered["match_date"],
        xi=config.time_decay_xi,
        reference_date=(
            pd.Timestamp(config.training_cutoff)
            if config.training_cutoff is not None
            else pd.Timestamp(filtered["match_date"].max())
        ),
    )
    return filtered.reset_index(drop=True)


def train_baseline_model(
    results: pd.DataFrame,
    *,
    config: BaselineTrainingConfig | None = None,
) -> tuple[DixonColesModel, pd.DataFrame]:
    config = config or BaselineTrainingConfig()
    training_frame = build_training_frame(results, config=config)
    model = DixonColesModel.fit(
        training_frame,
        l2_penalty=config.l2_penalty,
        maxiter=config.maxiter,
    )
    return model, training_frame
