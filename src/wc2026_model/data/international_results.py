from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from wc2026_model.types import OUTCOME_AWAY, OUTCOME_DRAW, OUTCOME_HOME

DEFAULT_INTERNATIONAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

_REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
}

_STANDARDIZED_REQUIRED_COLUMNS = {
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "tournament",
    "neutral",
}

_TOURNAMENT_CANONICALIZATION_RULES = (
    ("world cup - qualification", "FIFA World Cup qualification"),
    ("fifa world cup qualification", "FIFA World Cup qualification"),
    ("euro championship - qualification", "UEFA Euro qualification"),
    ("uefa euro qualification", "UEFA Euro qualification"),
    ("euro championship", "UEFA Euro"),
    ("uefa euro", "UEFA Euro"),
    ("friendlies", "Friendly"),
    ("friendly", "Friendly"),
    ("unofficial friendly", "Unofficial Friendly"),
    ("world cup", "FIFA World Cup"),
    ("copa america", "Copa America"),
    ("africa cup of nations qualification", "African Cup of Nations qualification"),
    ("africa cup of nations", "African Cup of Nations"),
    ("african cup of nations qualification", "African Cup of Nations qualification"),
    ("african cup of nations", "African Cup of Nations"),
    ("concacaf gold cup", "Gold Cup"),
    ("gold cup", "Gold Cup"),
    ("uefa nations league", "UEFA Nations League"),
    ("concacaf nations league", "CONCACAF Nations League"),
    ("arab cup", "Arab Cup"),
    ("asian cup", "AFC Asian Cup"),
)

_TEAM_NAME_ALIASES = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo, DR": "DR Congo",
    "Congo DR": "DR Congo",
    "Cote d'Ivoire": "Ivory Coast",
    "Curacao": "Curaçao",
    "CuraÃ§ao": "Curaçao",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "Democratic Republic of the Congo": "DR Congo",
    "French Guyana": "French Guiana",
    "FYR Macedonia": "North Macedonia",
    "Korea DPR": "North Korea",
    "Korea, Republic of": "South Korea",
    "Korea Republic": "South Korea",
    "Rep. Of Ireland": "Republic of Ireland",
    "Republic of Korea": "South Korea",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "TÃ¼rkiye": "Turkey",
    "Türkiye": "Turkey",
    "United States of America": "United States",
    "USA": "United States",
}


def download_international_results_csv(
    destination: str | Path,
    *,
    source_url: str = DEFAULT_INTERNATIONAL_RESULTS_URL,
    overwrite: bool = False,
) -> Path:
    destination_path = Path(destination)
    if destination_path.exists() and not overwrite:
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(source_url) as response, destination_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)
    return destination_path


def load_international_results(path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path, low_memory=False)
    if _REQUIRED_COLUMNS.issubset(dataframe.columns):
        return standardize_international_results(dataframe)
    if _STANDARDIZED_REQUIRED_COLUMNS.issubset(dataframe.columns):
        return normalize_standardized_results(dataframe)

    missing_raw = _REQUIRED_COLUMNS.difference(dataframe.columns)
    missing_standardized = _STANDARDIZED_REQUIRED_COLUMNS.difference(dataframe.columns)
    raise ValueError(
        "Input data is missing both the raw and standardized required columns. "
        f"Missing raw columns: {', '.join(sorted(missing_raw))}. "
        f"Missing standardized columns: {', '.join(sorted(missing_standardized))}."
    )


def load_scheduled_matches(
    path: str | Path,
    *,
    tournament: str | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    dataframe = pd.read_csv(path, low_memory=False)
    missing_columns = _REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in raw results/fixtures data: {missing}")

    fixtures = dataframe.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="raise")
    fixtures["tournament"] = fixtures["tournament"].map(canonicalize_tournament_name)
    fixtures["home_team"] = fixtures["home_team"].map(canonicalize_team_name)
    fixtures["away_team"] = fixtures["away_team"].map(canonicalize_team_name)
    fixtures["neutral"] = fixtures["neutral"].astype(bool)

    fixtures = fixtures[fixtures["home_score"].isna() & fixtures["away_score"].isna()].copy()
    if tournament is not None:
        fixtures = fixtures[fixtures["tournament"] == canonicalize_tournament_name(tournament)]
    if start_date is not None:
        fixtures = fixtures[fixtures["date"] >= pd.Timestamp(start_date)]

    fixtures = fixtures.rename(columns={"date": "match_date"})
    fixtures["match_id"] = (
        fixtures["match_date"].dt.strftime("%Y-%m-%d")
        + "::"
        + fixtures["home_team"]
        + "::"
        + fixtures["away_team"]
    )
    return fixtures.sort_values(["match_date", "home_team", "away_team"], kind="stable").reset_index(
        drop=True
    )


def standardize_international_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing_columns = _REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in international results data: {missing}")

    standardized = dataframe.copy()
    standardized["date"] = pd.to_datetime(standardized["date"], errors="raise")
    standardized["neutral"] = standardized["neutral"].astype(bool)
    standardized["tournament"] = standardized["tournament"].map(canonicalize_tournament_name)
    standardized["home_team"] = standardized["home_team"].map(canonicalize_team_name)
    standardized["away_team"] = standardized["away_team"].map(canonicalize_team_name)
    standardized = standardized[
        standardized["home_score"].notna() & standardized["away_score"].notna()
    ].copy()
    standardized["home_score"] = standardized["home_score"].astype(int)
    standardized["away_score"] = standardized["away_score"].astype(int)

    standardized = standardized.rename(
        columns={
            "date": "match_date",
            "home_score": "home_goals",
            "away_score": "away_goals",
        }
    )
    return normalize_standardized_results(standardized)


def normalize_standardized_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing_columns = _STANDARDIZED_REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required standardized columns: {missing}")

    standardized = dataframe.copy()
    standardized["match_date"] = pd.to_datetime(standardized["match_date"], errors="raise")
    standardized["neutral"] = standardized["neutral"].astype(bool)
    standardized["tournament"] = standardized["tournament"].map(canonicalize_tournament_name)
    standardized["home_team"] = standardized["home_team"].map(canonicalize_team_name)
    standardized["away_team"] = standardized["away_team"].map(canonicalize_team_name)
    standardized = standardized[
        standardized["home_goals"].notna() & standardized["away_goals"].notna()
    ].copy()
    standardized["home_goals"] = standardized["home_goals"].astype(int)
    standardized["away_goals"] = standardized["away_goals"].astype(int)

    if "city" not in standardized.columns:
        standardized["city"] = None
    if "country" not in standardized.columns:
        standardized["country"] = None

    standardized["goal_diff"] = standardized["home_goals"] - standardized["away_goals"]
    standardized["total_goals"] = standardized["home_goals"] + standardized["away_goals"]
    standardized["home_result"] = np.select(
        [
            standardized["home_goals"] > standardized["away_goals"],
            standardized["home_goals"] == standardized["away_goals"],
        ],
        [OUTCOME_HOME, OUTCOME_DRAW],
        default=OUTCOME_AWAY,
    )
    standardized["is_competitive"] = standardized["tournament"].map(is_competitive_tournament)
    standardized["home_points"] = np.select(
        [
            standardized["home_goals"] > standardized["away_goals"],
            standardized["home_goals"] == standardized["away_goals"],
        ],
        [3, 1],
        default=0,
    )
    standardized["away_points"] = np.select(
        [
            standardized["away_goals"] > standardized["home_goals"],
            standardized["away_goals"] == standardized["home_goals"],
        ],
        [3, 1],
        default=0,
    )
    standardized["match_id"] = (
        standardized["match_date"].dt.strftime("%Y-%m-%d")
        + "::"
        + standardized["home_team"]
        + "::"
        + standardized["away_team"]
    )
    standardized = standardized.sort_values(
        ["match_date", "home_team", "away_team"], kind="stable"
    ).reset_index(drop=True)
    return standardized


def canonicalize_tournament_name(tournament_name: str) -> str:
    normalized = str(tournament_name).strip()
    normalized_lower = normalized.lower()
    for pattern, replacement in _TOURNAMENT_CANONICALIZATION_RULES:
        if normalized_lower == pattern or normalized_lower.startswith(f"{pattern} "):
            return replacement
    return normalized


def canonicalize_team_name(team_name: str) -> str:
    normalized = str(team_name).strip()
    return _TEAM_NAME_ALIASES.get(normalized, normalized)


def is_competitive_tournament(tournament_name: str) -> bool:
    normalized = canonicalize_tournament_name(tournament_name).lower()
    if "friendly" in normalized:
        return False
    if normalized in {"friendly", "unofficial friendly"}:
        return False
    return True
