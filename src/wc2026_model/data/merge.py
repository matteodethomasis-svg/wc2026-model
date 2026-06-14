from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

# Different sources log the same fixture up to a day apart (timezone / record date),
# so exact-date dedupe misses them and the match is counted twice in the Elo. Treat
# same-teams + same-score records within this window as the same match.
DEFAULT_DUPLICATE_DATE_TOLERANCE_DAYS = 2


def combine_standardized_results(
    frames: Iterable[pd.DataFrame],
    *,
    source_priority: tuple[str, ...] = ("football_data_api", "cup26_open", "historical_csv"),
    duplicate_date_tolerance_days: int = DEFAULT_DUPLICATE_DATE_TOLERANCE_DAYS,
) -> pd.DataFrame:
    frame_list = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not frame_list:
        return pd.DataFrame()

    combined = pd.concat(frame_list, ignore_index=True, sort=False)
    if "source" not in combined.columns:
        combined["source"] = "unknown"
    # Sources may supply match_date as strings or Timestamps; normalize so sorting and
    # near-date dedupe compare like-with-like (mixed str/Timestamp raises on sort).
    combined["match_date"] = pd.to_datetime(combined["match_date"], errors="coerce")

    priority_rank = {source_name: rank for rank, source_name in enumerate(source_priority)}
    combined["_source_priority"] = combined["source"].map(priority_rank).fillna(len(priority_rank))
    combined = combined.sort_values(
        ["match_date", "home_team", "away_team", "_source_priority"],
        kind="stable",
    )
    # First drop exact duplicates (same date), keeping the higher-priority source.
    dedupe_columns = ["match_date", "home_team", "away_team", "home_goals", "away_goals"]
    combined = combined.drop_duplicates(subset=dedupe_columns, keep="first")
    # Then drop cross-date duplicates of the same fixture (different source, ±tolerance).
    combined = _drop_near_date_duplicates(
        combined, tolerance_days=duplicate_date_tolerance_days
    )
    combined = combined.drop(columns="_source_priority")
    return combined.sort_values(["match_date", "home_team", "away_team"], kind="stable").reset_index(
        drop=True
    )


def _drop_near_date_duplicates(
    combined: pd.DataFrame,
    *,
    tolerance_days: int,
) -> pd.DataFrame:
    if combined.empty or tolerance_days <= 0:
        return combined

    parsed_date = pd.to_datetime(combined["match_date"], errors="coerce")
    # Within each (teams, score) group, walk records in date order and drop any whose
    # date is within tolerance of an already-kept record. Higher-priority source first
    # (stable sort) so the kept record is the preferred one.
    order = combined.assign(parsed_match_date=parsed_date).sort_values(
        ["home_team", "away_team", "home_goals", "away_goals", "_source_priority", "parsed_match_date"],
        kind="stable",
    )
    drop_index: list = []
    kept_dates: dict[tuple, list[pd.Timestamp]] = {}
    for row in order.itertuples():
        key = (row.home_team, row.away_team, row.home_goals, row.away_goals)
        match_date = row.parsed_match_date
        if pd.isna(match_date):
            continue
        seen = kept_dates.setdefault(key, [])
        if any(abs((match_date - kept).days) <= tolerance_days for kept in seen):
            drop_index.append(row.Index)
        else:
            seen.append(match_date)
    return combined.drop(index=drop_index)
