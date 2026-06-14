from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecentFormConfig:
    window_size: int = 5


def augment_with_pre_match_form_features(
    matches: pd.DataFrame,
    *,
    config: RecentFormConfig | None = None,
) -> pd.DataFrame:
    config = config or RecentFormConfig()
    if config.window_size <= 0:
        raise ValueError(f"window_size must be positive, got {config.window_size}.")

    required_columns = {
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    }
    missing_columns = required_columns.difference(matches.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns for form augmentation: {missing}")

    dataframe = matches.copy()
    dataframe["match_date"] = pd.to_datetime(dataframe["match_date"], errors="raise")
    dataframe = dataframe.sort_values(
        ["match_date", "home_team", "away_team"], kind="stable"
    ).reset_index(drop=True)

    histories: dict[str, Deque[dict[str, float | pd.Timestamp]]] = defaultdict(
        lambda: deque(maxlen=config.window_size)
    )

    home_records: list[dict[str, float]] = []
    away_records: list[dict[str, float]] = []
    for row in dataframe.itertuples(index=False):
        match_date = pd.Timestamp(row.match_date)
        home_team = str(row.home_team)
        away_team = str(row.away_team)

        home_records.append(_summarize_team_history(histories.get(home_team), match_date))
        away_records.append(_summarize_team_history(histories.get(away_team), match_date))

        _append_team_match(
            histories[home_team],
            match_date=match_date,
            goals_for=float(row.home_goals),
            goals_against=float(row.away_goals),
        )
        _append_team_match(
            histories[away_team],
            match_date=match_date,
            goals_for=float(row.away_goals),
            goals_against=float(row.home_goals),
        )

    home_frame = pd.DataFrame.from_records(home_records).add_prefix("home_")
    away_frame = pd.DataFrame.from_records(away_records).add_prefix("away_")
    return pd.concat([dataframe, home_frame, away_frame], axis=1)


def _summarize_team_history(
    history: Deque[dict[str, float | pd.Timestamp]] | None,
    match_date: pd.Timestamp,
) -> dict[str, float]:
    if not history:
        return {
            "form_match_count": 0.0,
            "form_points_per_match": np.nan,
            "form_goal_diff_per_match": np.nan,
            "form_goals_for_per_match": np.nan,
            "form_goals_against_per_match": np.nan,
            "form_win_rate": np.nan,
            "days_since_last_match": np.nan,
        }

    count = float(len(history))
    points = np.array([float(item["points"]) for item in history], dtype=float)
    goal_diff = np.array([float(item["goal_diff"]) for item in history], dtype=float)
    goals_for = np.array([float(item["goals_for"]) for item in history], dtype=float)
    goals_against = np.array([float(item["goals_against"]) for item in history], dtype=float)
    wins = np.array([float(item["win"]) for item in history], dtype=float)
    last_match_date = pd.Timestamp(history[-1]["match_date"])

    return {
        "form_match_count": count,
        "form_points_per_match": float(points.mean()),
        "form_goal_diff_per_match": float(goal_diff.mean()),
        "form_goals_for_per_match": float(goals_for.mean()),
        "form_goals_against_per_match": float(goals_against.mean()),
        "form_win_rate": float(wins.mean()),
        "days_since_last_match": float((match_date - last_match_date).days),
    }


def _append_team_match(
    history: Deque[dict[str, float | pd.Timestamp]],
    *,
    match_date: pd.Timestamp,
    goals_for: float,
    goals_against: float,
) -> None:
    if goals_for > goals_against:
        points = 3.0
        win = 1.0
    elif goals_for == goals_against:
        points = 1.0
        win = 0.0
    else:
        points = 0.0
        win = 0.0

    history.append(
        {
            "match_date": match_date,
            "points": points,
            "goal_diff": goals_for - goals_against,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "win": win,
        }
    )
