from __future__ import annotations

from pathlib import Path

import pandas as pd
import soccerdata as sd


def load_club_elo_snapshot(
    *,
    as_of_date: str | None = None,
    input_path: str | Path | None = None,
    proxy: str | None = None,
    no_cache: bool = False,
    no_store: bool = False,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    if input_path is not None:
        frame = pd.read_csv(input_path)
        return _standardize_club_elo_snapshot(frame)

    reader_kwargs = {
        "proxy": proxy,
        "no_cache": no_cache,
        "no_store": no_store,
    }
    if data_dir is not None:
        reader_kwargs["data_dir"] = Path(data_dir)
    reader = sd.ClubElo(**reader_kwargs)
    frame = reader.read_by_date(as_of_date).reset_index()
    return _standardize_club_elo_snapshot(frame)


def save_club_elo_snapshot_csv(
    destination: str | Path,
    *,
    as_of_date: str | None = None,
    proxy: str | None = None,
    no_cache: bool = False,
    no_store: bool = False,
    data_dir: str | Path | None = None,
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    frame = load_club_elo_snapshot(
        as_of_date=as_of_date,
        proxy=proxy,
        no_cache=no_cache,
        no_store=no_store,
        data_dir=data_dir,
    )
    frame.to_csv(destination_path, index=False)
    return destination_path


def _standardize_club_elo_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    standardized = frame.copy()
    standardized = standardized.rename(
        columns={
            "team": "club",
            "country": "club_country",
            "elo": "club_elo",
            "league": "club_league",
            "from": "rating_from",
            "to": "rating_to",
        }
    )
    required_columns = {"club", "club_elo"}
    missing_columns = required_columns.difference(standardized.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required Club Elo columns: {missing}")

    standardized["club"] = standardized["club"].astype(str)
    standardized["club_elo"] = pd.to_numeric(standardized["club_elo"], errors="coerce")
    for column in ("rating_from", "rating_to"):
        if column in standardized.columns:
            standardized[column] = pd.to_datetime(standardized[column], errors="coerce")
    return standardized.sort_values("club_elo", ascending=False, kind="stable").reset_index(
        drop=True
    )
