from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from wc2026_model.data import load_international_results

from .simulation import build_group_stage_schedule


def load_played_group_results(
    path: str | Path,
    *,
    groups: Mapping[str, Sequence[str]],
    as_of_date: str | None = None,
    tournament_year: int | None = 2026,
) -> pd.DataFrame:
    results = load_international_results(path)
    if as_of_date is not None:
        results = results.loc[results["match_date"] <= pd.Timestamp(as_of_date)].copy()

    world_cup_results = results.loc[results["tournament"] == "FIFA World Cup"].copy()
    if tournament_year is not None:
        world_cup_results = world_cup_results.loc[
            world_cup_results["match_date"].dt.year == int(tournament_year)
        ].copy()
    if world_cup_results.empty:
        return pd.DataFrame(columns=["group", "home_team", "away_team", "home_goals", "away_goals"])

    schedule = build_group_stage_schedule(groups).loc[:, ["group", "home_team", "away_team"]].copy()
    played_group_results = world_cup_results.merge(
        schedule,
        on=["home_team", "away_team"],
        how="inner",
    )
    return (
        played_group_results.loc[
            :, ["group", "home_team", "away_team", "home_goals", "away_goals"]
        ]
        .drop_duplicates(subset=["group", "home_team", "away_team"], keep="last")
        .reset_index(drop=True)
    )
