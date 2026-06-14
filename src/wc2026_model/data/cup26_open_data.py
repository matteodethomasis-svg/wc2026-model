from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from .international_results import canonicalize_tournament_name, standardize_international_results

DEFAULT_CUP26_OPEN_RESULTS_URL = (
    "https://raw.githubusercontent.com/Hicruben/world-cup-2026-prediction-model/main/data/results.json"
)


def download_cup26_open_results_json(
    destination: str | Path,
    *,
    source_url: str = DEFAULT_CUP26_OPEN_RESULTS_URL,
    overwrite: bool = False,
) -> Path:
    destination_path = Path(destination)
    if destination_path.exists() and not overwrite:
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(source_url) as response, destination_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)
    return destination_path


def load_cup26_open_results(path: str | Path) -> pd.DataFrame:
    with Path(path).open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    return standardize_cup26_open_results(payload)


def standardize_cup26_open_results(payload: dict[str, object]) -> pd.DataFrame:
    matches = payload.get("matches")
    if not isinstance(matches, list):
        raise ValueError("Cup26 open data payload must contain a list under 'matches'.")

    raw_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
        "source_match_id",
        "source_competition_id",
        "source_competition_name",
    ]
    raw_rows = []
    for match in matches:
        if not isinstance(match, dict):
            raise ValueError("Cup26 open data matches must be JSON objects.")
        raw_rows.append(
            {
                "date": match["date"],
                "home_team": match["homeName"],
                "away_team": match["awayName"],
                "home_score": match["hg"],
                "away_score": match["ag"],
                "tournament": canonicalize_tournament_name(str(match["leagueName"])),
                "city": None,
                "country": None,
                "neutral": infer_neutral_site_from_tournament(str(match["leagueName"])),
                "source_match_id": match.get("id"),
                "source_competition_id": match.get("leagueId"),
                "source_competition_name": match.get("leagueName"),
            }
        )

    standardized = standardize_international_results(
        pd.DataFrame.from_records(raw_rows, columns=raw_columns)
    )
    standardized["source"] = "cup26_open"
    standardized["source_generated_at_utc"] = payload.get("generatedAt")
    return standardized


def infer_neutral_site_from_tournament(tournament_name: str) -> bool:
    canonical_name = canonicalize_tournament_name(tournament_name)
    return canonical_name in {
        "FIFA World Cup",
        "UEFA Euro",
        "Copa America",
        "African Cup of Nations",
        "Gold Cup",
        "AFC Asian Cup",
        "Arab Cup",
    }
