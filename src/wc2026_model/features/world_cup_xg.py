from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WorldCupXGConfig:
    window_size: int = 5


def augment_with_pre_match_xg_features(
    matches: pd.DataFrame,
    *,
    config: WorldCupXGConfig | None = None,
) -> pd.DataFrame:
    config = config or WorldCupXGConfig()
    if config.window_size <= 0:
        raise ValueError(f"window_size must be positive, got {config.window_size}.")

    required_columns = {
        "match_date",
        "home_team",
        "away_team",
        "home_xg",
        "away_xg",
    }
    missing_columns = required_columns.difference(matches.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns for xG augmentation: {missing}")

    column_flags = _detect_optional_columns(matches)

    dataframe = matches.copy()
    dataframe["match_date"] = pd.to_datetime(dataframe["match_date"], errors="raise")
    dataframe = dataframe.sort_values(
        ["match_date", "home_team", "away_team"],
        kind="stable",
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
            **_match_stats_for_side(row, side="home", column_flags=column_flags),
        )
        _append_team_match(
            histories[away_team],
            match_date=match_date,
            **_match_stats_for_side(row, side="away", column_flags=column_flags),
        )

    home_frame = pd.DataFrame.from_records(home_records).add_prefix("home_")
    away_frame = pd.DataFrame.from_records(away_records).add_prefix("away_")
    return pd.concat([dataframe, home_frame, away_frame], axis=1)


def build_latest_team_xg_snapshot(
    matches: pd.DataFrame,
    *,
    config: WorldCupXGConfig | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    config = config or WorldCupXGConfig()
    if config.window_size <= 0:
        raise ValueError(f"window_size must be positive, got {config.window_size}.")

    required_columns = {
        "match_date",
        "home_team",
        "away_team",
        "home_xg",
        "away_xg",
    }
    missing_columns = required_columns.difference(matches.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns for xG snapshot: {missing}")

    column_flags = _detect_optional_columns(matches)

    dataframe = matches.copy()
    dataframe["match_date"] = pd.to_datetime(dataframe["match_date"], errors="raise")
    if as_of_date is not None:
        reference_date = pd.Timestamp(as_of_date)
        dataframe = dataframe[dataframe["match_date"] < reference_date].copy()
    else:
        reference_date = (
            pd.Timestamp(dataframe["match_date"].max()) if not dataframe.empty else pd.Timestamp.utcnow()
        )

    dataframe = dataframe.sort_values(
        ["match_date", "home_team", "away_team"],
        kind="stable",
    ).reset_index(drop=True)

    histories: dict[str, Deque[dict[str, float | pd.Timestamp]]] = defaultdict(
        lambda: deque(maxlen=config.window_size)
    )
    for row in dataframe.itertuples(index=False):
        _append_team_match(
            histories[str(row.home_team)],
            match_date=pd.Timestamp(row.match_date),
            **_match_stats_for_side(row, side="home", column_flags=column_flags),
        )
        _append_team_match(
            histories[str(row.away_team)],
            match_date=pd.Timestamp(row.match_date),
            **_match_stats_for_side(row, side="away", column_flags=column_flags),
        )

    rows: list[dict[str, float | str]] = []
    for team, history in sorted(histories.items()):
        if not history:
            continue
        rows.append({"team": team} | _summarize_team_history(history, reference_date))
    return pd.DataFrame.from_records(rows)


def attach_latest_team_xg_features(
    fixtures: pd.DataFrame,
    team_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    required_fixture_columns = {"home_team", "away_team"}
    missing_fixture_columns = required_fixture_columns.difference(fixtures.columns)
    if missing_fixture_columns:
        missing = ", ".join(sorted(missing_fixture_columns))
        raise ValueError(f"Missing required fixture columns: {missing}")
    if team_snapshot.empty:
        return fixtures.copy()
    if "team" not in team_snapshot.columns:
        raise ValueError("Team snapshot must contain a 'team' column.")

    snapshot_columns = [column for column in team_snapshot.columns if column != "team"]
    home_snapshot = team_snapshot.rename(
        columns={column: f"home_{column}" for column in snapshot_columns}
    )
    away_snapshot = team_snapshot.rename(
        columns={column: f"away_{column}" for column in snapshot_columns}
    )
    enriched = fixtures.copy()
    enriched = enriched.merge(
        home_snapshot,
        left_on="home_team",
        right_on="team",
        how="left",
    ).drop(columns=["team"])
    enriched = enriched.merge(
        away_snapshot,
        left_on="away_team",
        right_on="team",
        how="left",
    ).drop(columns=["team"])
    return enriched


def _summarize_team_history(
    history: Deque[dict[str, float | pd.Timestamp]] | None,
    match_date: pd.Timestamp,
) -> dict[str, float]:
    if not history:
        return {
            "xg_match_count": 0.0,
            "xg_for_per_match": np.nan,
            "xg_against_per_match": np.nan,
            "xg_diff_per_match": np.nan,
            "shots_for_per_match": np.nan,
            "shots_against_per_match": np.nan,
            "shots_on_target_for_per_match": np.nan,
            "shots_on_target_against_per_match": np.nan,
            "shot_accuracy_for": np.nan,
            "shot_accuracy_against": np.nan,
            "xg_per_shot": np.nan,
            "passes_for_per_match": np.nan,
            "passes_against_per_match": np.nan,
            "pass_completion_for": np.nan,
            "pass_completion_against": np.nan,
            "pressures_for_per_match": np.nan,
            "pressures_against_per_match": np.nan,
            "days_since_last_xg_match": np.nan,
        }

    count = float(len(history))
    xg_for = np.array([float(item["xg_for"]) for item in history], dtype=float)
    xg_against = np.array([float(item["xg_against"]) for item in history], dtype=float)
    shots_for = np.array([float(item["shots_for"]) for item in history], dtype=float)
    shots_against = np.array([float(item["shots_against"]) for item in history], dtype=float)
    shots_on_target_for = np.array(
        [float(item["shots_on_target_for"]) for item in history],
        dtype=float,
    )
    shots_on_target_against = np.array(
        [float(item["shots_on_target_against"]) for item in history],
        dtype=float,
    )
    passes_for = _history_array(history, "passes_for")
    completed_passes_for = _history_array(history, "completed_passes_for")
    passes_against = _history_array(history, "passes_against")
    completed_passes_against = _history_array(history, "completed_passes_against")
    pressures_for = _history_array(history, "pressures_for")
    pressures_against = _history_array(history, "pressures_against")
    last_match_date = pd.Timestamp(history[-1]["match_date"])

    mean_xg_for = float(xg_for.mean())
    mean_xg_against = float(xg_against.mean())
    mean_shots_for = _nanmean_or_nan(shots_for)
    mean_shots_against = _nanmean_or_nan(shots_against)
    mean_shots_on_target_for = _nanmean_or_nan(shots_on_target_for)
    mean_shots_on_target_against = _nanmean_or_nan(shots_on_target_against)

    return {
        "xg_match_count": count,
        "xg_for_per_match": mean_xg_for,
        "xg_against_per_match": mean_xg_against,
        "xg_diff_per_match": mean_xg_for - mean_xg_against,
        "shots_for_per_match": mean_shots_for,
        "shots_against_per_match": mean_shots_against,
        "shots_on_target_for_per_match": mean_shots_on_target_for,
        "shots_on_target_against_per_match": mean_shots_on_target_against,
        "shot_accuracy_for": _safe_rate(
            numerator=mean_shots_on_target_for,
            denominator=mean_shots_for,
        ),
        "shot_accuracy_against": _safe_rate(
            numerator=mean_shots_on_target_against,
            denominator=mean_shots_against,
        ),
        "xg_per_shot": _safe_rate(numerator=mean_xg_for, denominator=mean_shots_for),
        "passes_for_per_match": _nanmean_or_nan(passes_for),
        "passes_against_per_match": _nanmean_or_nan(passes_against),
        "pass_completion_for": _safe_rate(
            numerator=_nansum_or_nan(completed_passes_for),
            denominator=_nansum_or_nan(passes_for),
        ),
        "pass_completion_against": _safe_rate(
            numerator=_nansum_or_nan(completed_passes_against),
            denominator=_nansum_or_nan(passes_against),
        ),
        "pressures_for_per_match": _nanmean_or_nan(pressures_for),
        "pressures_against_per_match": _nanmean_or_nan(pressures_against),
        "days_since_last_xg_match": float((match_date - last_match_date).days),
    }


def _append_team_match(
    history: Deque[dict[str, float | pd.Timestamp]],
    *,
    match_date: pd.Timestamp,
    xg_for: float,
    xg_against: float,
    shots_for: float,
    shots_against: float,
    shots_on_target_for: float,
    shots_on_target_against: float,
    passes_for: float = float("nan"),
    completed_passes_for: float = float("nan"),
    passes_against: float = float("nan"),
    completed_passes_against: float = float("nan"),
    pressures_for: float = float("nan"),
    pressures_against: float = float("nan"),
) -> None:
    history.append(
        {
            "match_date": match_date,
            "xg_for": xg_for,
            "xg_against": xg_against,
            "shots_for": shots_for,
            "shots_against": shots_against,
            "shots_on_target_for": shots_on_target_for,
            "shots_on_target_against": shots_on_target_against,
            "passes_for": passes_for,
            "completed_passes_for": completed_passes_for,
            "passes_against": passes_against,
            "completed_passes_against": completed_passes_against,
            "pressures_for": pressures_for,
            "pressures_against": pressures_against,
        }
    )


def _detect_optional_columns(matches: pd.DataFrame) -> dict[str, bool]:
    columns = matches.columns
    return {
        "shots": {"home_shots", "away_shots"}.issubset(columns),
        "shots_on_target": {
            "home_shots_on_target",
            "away_shots_on_target",
        }.issubset(columns),
        "passes": {
            "home_passes",
            "away_passes",
            "home_completed_passes",
            "away_completed_passes",
        }.issubset(columns),
        "pressures": {"home_pressures", "away_pressures"}.issubset(columns),
    }


def _match_stats_for_side(
    row: object,
    *,
    side: str,
    column_flags: dict[str, bool],
) -> dict[str, float]:
    opponent = "away" if side == "home" else "home"

    def value(prefix: str, field: str, *, available: bool) -> float:
        if not available:
            return float("nan")
        return float(getattr(row, f"{prefix}_{field}"))

    return {
        "xg_for": float(getattr(row, f"{side}_xg")),
        "xg_against": float(getattr(row, f"{opponent}_xg")),
        "shots_for": value(side, "shots", available=column_flags["shots"]),
        "shots_against": value(opponent, "shots", available=column_flags["shots"]),
        "shots_on_target_for": value(
            side, "shots_on_target", available=column_flags["shots_on_target"]
        ),
        "shots_on_target_against": value(
            opponent, "shots_on_target", available=column_flags["shots_on_target"]
        ),
        "passes_for": value(side, "passes", available=column_flags["passes"]),
        "completed_passes_for": value(
            side, "completed_passes", available=column_flags["passes"]
        ),
        "passes_against": value(opponent, "passes", available=column_flags["passes"]),
        "completed_passes_against": value(
            opponent, "completed_passes", available=column_flags["passes"]
        ),
        "pressures_for": value(side, "pressures", available=column_flags["pressures"]),
        "pressures_against": value(
            opponent, "pressures", available=column_flags["pressures"]
        ),
    }


def _history_array(
    history: Deque[dict[str, float | pd.Timestamp]],
    field: str,
) -> np.ndarray:
    return np.array([float(item.get(field, np.nan)) for item in history], dtype=float)


def _nanmean_or_nan(values: np.ndarray) -> float:
    if np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


def _nansum_or_nan(values: np.ndarray) -> float:
    if np.isnan(values).all():
        return float("nan")
    return float(np.nansum(values))


def _safe_rate(*, numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return numerator / denominator
